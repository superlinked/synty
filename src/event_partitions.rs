// Event-time coverage for physical raw-event partitions. Trackers publish one
// small mutable index per owned stream; Athena readers use it to choose the
// capture-day directories that can contain a requested event-time window.
// Missing or legacy coverage stays conservative: include the physical day
// rather than risk a false negative.

use crate::bucket::Bucket;
use anyhow::{Result, anyhow};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const PREFIX: &str = "event-partitions";
const FORMAT: u32 = 1;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct EventTimeRange {
    pub min_ts: String,
    pub max_ts: String,
}

impl EventTimeRange {
    fn record(&mut self, ts: &str) {
        if self.min_ts.is_empty() || ts < self.min_ts.as_str() {
            self.min_ts = ts.to_string();
        }
        if self.max_ts.is_empty() || ts > self.max_ts.as_str() {
            self.max_ts = ts.to_string();
        }
    }

    #[cfg_attr(not(feature = "athena"), allow(dead_code))]
    fn overlaps(&self, since: DateTime<Utc>, until: DateTime<Utc>) -> bool {
        let Some(min) = parse_time(&self.min_ts) else {
            return true;
        };
        let Some(max) = parse_time(&self.max_ts) else {
            return true;
        };
        max >= since && min < until
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct EventPartitionIndex {
    #[serde(default = "format")]
    pub format: u32,
    pub stream: String,
    /// Physical days whose exact event-time coverage has not been initialized.
    /// Readers always include them, preserving correctness for old writers.
    #[serde(default)]
    pub legacy_days: BTreeSet<String>,
    /// Physical day → inclusive event-time range for every object in that day.
    #[serde(default)]
    pub partitions: BTreeMap<String, EventTimeRange>,
    #[serde(default)]
    pub updated_at: String,
}

fn format() -> u32 {
    FORMAT
}

impl EventPartitionIndex {
    fn empty(stream: &str) -> Self {
        Self {
            format: FORMAT,
            stream: stream.to_string(),
            legacy_days: BTreeSet::new(),
            partitions: BTreeMap::new(),
            updated_at: String::new(),
        }
    }

    pub(crate) fn physical_days(&self) -> BTreeSet<String> {
        self.legacy_days
            .iter()
            .chain(self.partitions.keys())
            .cloned()
            .collect()
    }

    #[cfg_attr(not(feature = "athena"), allow(dead_code))]
    pub(crate) fn candidate_days(
        &self,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
    ) -> BTreeSet<String> {
        let mut days = self.legacy_days.clone();
        days.extend(
            self.partitions
                .iter()
                .filter(|(_, range)| range.overlaps(since, until))
                .map(|(day, _)| day.clone()),
        );
        days
    }

    pub(crate) fn newest_event(&self) -> Option<String> {
        self.partitions
            .values()
            .filter_map(|range| parse_time(&range.max_ts))
            .max()
            .map(|timestamp| timestamp.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true))
    }

    /// Mark a historical physical day as completely indexed. This is the
    /// metadata-only backfill seam: callers may derive the range with Athena,
    /// then remove the conservative legacy marker without moving raw objects.
    #[cfg(test)]
    fn complete_legacy_day(&mut self, day: &str, min_ts: &str, max_ts: &str) {
        self.legacy_days.remove(day);
        self.partitions.insert(
            day.to_string(),
            EventTimeRange {
                min_ts: min_ts.to_string(),
                max_ts: max_ts.to_string(),
            },
        );
    }
}

pub(crate) fn key(stream: &str) -> String {
    format!("{PREFIX}/{stream}.json")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct EventObjectIndex {
    #[serde(default = "format")]
    pub format: u32,
    pub stream: String,
    pub day: String,
    #[serde(default)]
    pub objects: BTreeMap<String, EventTimeRange>,
    #[serde(default)]
    pub updated_at: String,
}

impl EventObjectIndex {
    fn empty(stream: &str, day: &str) -> Self {
        Self {
            format: FORMAT,
            stream: stream.to_string(),
            day: day.to_string(),
            objects: BTreeMap::new(),
            updated_at: String::new(),
        }
    }

    #[cfg_attr(not(feature = "athena"), allow(dead_code))]
    pub(crate) fn candidate_objects(
        &self,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
    ) -> BTreeSet<String> {
        self.objects
            .iter()
            .filter(|(_, range)| range.overlaps(since, until))
            .map(|(key, _)| key.clone())
            .collect()
    }
}

pub(crate) fn object_key(stream: &str, day: &str) -> String {
    format!("{PREFIX}/{stream}/track.{day}.json")
}

pub(crate) fn load_objects(
    bucket: &dyn Bucket,
    stream: &str,
    day: &str,
) -> Result<Option<EventObjectIndex>> {
    let Some(raw) = bucket.get(&object_key(stream, day))? else {
        return Ok(None);
    };
    let index: EventObjectIndex = serde_json::from_slice(&raw)
        .map_err(|error| anyhow!("invalid event object index for {stream}/{day}: {error}"))?;
    anyhow::ensure!(
        index.format == FORMAT && index.stream == stream && index.day == day,
        "event object index identity mismatch for {stream}/{day}"
    );
    Ok(Some(index))
}

pub(crate) fn load_objects_or_empty(
    bucket: &dyn Bucket,
    stream: &str,
    day: &str,
) -> Result<EventObjectIndex> {
    Ok(load_objects(bucket, stream, day)?.unwrap_or_else(|| EventObjectIndex::empty(stream, day)))
}

pub(crate) fn save_objects(bucket: &dyn Bucket, index: &EventObjectIndex) -> Result<()> {
    let mut index = index.clone();
    index.updated_at = Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    bucket.put(
        &object_key(&index.stream, &index.day),
        &serde_json::to_vec(&index)?,
    )
}

/// Record exact event-time coverage for one immutable JSONL object. Malformed
/// timestamps leave the object unindexed; readers discover it from the
/// physical listing and include it conservatively.
pub(crate) fn record_object(index: &mut EventObjectIndex, key: &str, raw: &[u8]) {
    let mut range = EventTimeRange::default();
    let mut records = 0usize;
    for line in raw.split(|byte| *byte == b'\n') {
        if line.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        let timestamp = serde_json::from_slice::<serde_json::Value>(line)
            .ok()
            .and_then(|value| value["ts"].as_str().and_then(parse_time));
        let Some(timestamp) = timestamp else {
            return;
        };
        range.record(&timestamp.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true));
        records += 1;
    }
    if records > 0 {
        index.objects.insert(key.to_string(), range);
    }
}

pub(crate) fn load(bucket: &dyn Bucket, stream: &str) -> Result<Option<EventPartitionIndex>> {
    let Some(raw) = bucket.get(&key(stream))? else {
        return Ok(None);
    };
    let index: EventPartitionIndex = serde_json::from_slice(&raw)
        .map_err(|error| anyhow!("invalid event partition index for {stream}: {error}"))?;
    anyhow::ensure!(
        index.format == FORMAT && index.stream == stream,
        "event partition index identity mismatch for {stream}"
    );
    Ok(Some(index))
}

/// Initialize before publishing event-day chunks. Every already-present day is
/// legacy and therefore always queried until an operator supplies its complete
/// min/max range. The object is written first, so a crash cannot make a new
/// layout silently look complete.
pub(crate) fn load_or_initialize(bucket: &dyn Bucket, stream: &str) -> Result<EventPartitionIndex> {
    if let Some(index) = load(bucket, stream)? {
        return Ok(index);
    }
    let mut index = EventPartitionIndex::empty(stream);
    for object in bucket.list(&format!("events/{stream}/chunks/"))? {
        if let Some(day) = day_from_event_key(&object) {
            index.legacy_days.insert(day.to_string());
        }
    }
    save(bucket, &index)?;
    Ok(index)
}

pub(crate) fn save(bucket: &dyn Bucket, index: &EventPartitionIndex) -> Result<()> {
    let mut index = index.clone();
    index.updated_at = Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    bucket.put(&key(&index.stream), &serde_json::to_vec(&index)?)
}

/// Newest indexed raw-event timestamp across the bucket. This reads only the
/// small per-stream metadata objects, never the JSONL event bodies.
pub(crate) fn bucket_newest_event(bucket: &dyn Bucket) -> Result<Option<String>> {
    let mut newest = None;
    for key in bucket.list(&format!("{PREFIX}/"))? {
        let Some(stream) = key
            .strip_prefix(&format!("{PREFIX}/"))
            .and_then(|name| name.strip_suffix(".json"))
            .filter(|stream| !stream.contains('/'))
        else {
            continue;
        };
        let Some(index) = load(bucket, stream)? else {
            continue;
        };
        if let Some(timestamp) = index.newest_event()
            && newest.as_deref().is_none_or(|current| timestamp.as_str() > current)
        {
            newest = Some(timestamp);
        }
    }
    Ok(newest)
}

/// Split complete JSONL lines by each event's UTC day and update the exact
/// range for those physical partitions. Unknown timestamps remain readable in
/// the source file's day, which is marked legacy so queries include it.
pub(crate) fn partition_lines(
    index: &mut EventPartitionIndex,
    raw: &[u8],
    fallback_day: &str,
) -> BTreeMap<String, Vec<u8>> {
    let mut by_day: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for line in raw.split_inclusive(|byte| *byte == b'\n') {
        if line.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        let timestamp = serde_json::from_slice::<serde_json::Value>(line)
            .ok()
            .and_then(|value| value["ts"].as_str().map(str::to_string));
        let (day, timestamp) = timestamp
            .as_deref()
            .and_then(parse_time)
            .map(|timestamp| {
                (
                    timestamp.format("%Y-%m-%d").to_string(),
                    Some(timestamp.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true)),
                )
            })
            .unwrap_or_else(|| (fallback_day.to_string(), None));
        by_day
            .entry(day.clone())
            .or_default()
            .extend_from_slice(line);
        if let Some(timestamp) = timestamp.as_deref() {
            index.partitions.entry(day).or_default().record(timestamp);
        } else {
            index.legacy_days.insert(day);
        }
    }
    by_day
}

pub(crate) fn day_from_event_key(key: &str) -> Option<&str> {
    let rest = key.split_once("/chunks/track.")?.1;
    let day = rest.split('/').next()?;
    valid_day(day).then_some(day)
}

pub(crate) fn day_from_track_file(file: &str) -> Option<&str> {
    let day = file.strip_prefix("track.")?.strip_suffix(".jsonl")?;
    valid_day(day).then_some(day)
}

fn valid_day(day: &str) -> bool {
    NaiveDate::parse_from_str(day, "%Y-%m-%d").is_ok()
}

fn parse_time(timestamp: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(timestamp)
        .ok()
        .map(|timestamp| timestamp.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(id: &str, ts: &str) -> String {
        serde_json::json!({"event_id": id, "ts": ts}).to_string() + "\n"
    }

    #[test]
    fn delayed_events_publish_under_their_utc_event_days() {
        let mut index = EventPartitionIndex::empty("edge-m-codex");
        let raw = event("old", "2026-07-22T23:55:00Z") + &event("new", "2026-07-24T01:00:00+01:00");
        let groups = partition_lines(&mut index, raw.as_bytes(), "2026-07-24");
        assert_eq!(
            groups.keys().cloned().collect::<Vec<_>>(),
            ["2026-07-22", "2026-07-24"]
        );
        assert_eq!(
            index.partitions["2026-07-22"].min_ts,
            "2026-07-22T23:55:00Z"
        );
    }

    #[test]
    fn unknown_timestamps_fail_open_to_a_conservative_physical_day() {
        let mut index = EventPartitionIndex::empty("edge-m-codex");
        let groups = partition_lines(&mut index, b"{\"event_id\":\"unknown\"}\n", "2026-07-24");
        assert!(groups.contains_key("2026-07-24"));
        assert!(index.legacy_days.contains("2026-07-24"));
    }

    #[test]
    fn partition_ranges_compare_normalized_instants_not_offset_strings() {
        let mut index = EventPartitionIndex::empty("edge-m-codex");
        let raw = event("late", "2026-07-22T12:30:00+02:00")
            + &event("early", "2026-07-22T10:00:00Z");
        partition_lines(&mut index, raw.as_bytes(), "2026-07-22");

        assert_eq!(
            index.partitions["2026-07-22"].min_ts,
            "2026-07-22T10:00:00Z"
        );
        assert_eq!(
            index.partitions["2026-07-22"].max_ts,
            "2026-07-22T10:30:00Z"
        );
    }

    #[test]
    fn object_ranges_prune_chunks_inside_a_broad_legacy_day() {
        let mut index = EventObjectIndex::empty("edge-m-codex", "2026-07-22");
        record_object(
            &mut index,
            "events/edge-m-codex/chunks/track.2026-07-22/old.jsonl",
            event("old", "2026-06-21T10:00:00Z").as_bytes(),
        );
        record_object(
            &mut index,
            "events/edge-m-codex/chunks/track.2026-07-22/new.jsonl",
            event("new", "2026-07-22T10:00:00Z").as_bytes(),
        );

        assert_eq!(
            index.candidate_objects(
                parse_time("2026-07-22T09:00:00Z").unwrap(),
                parse_time("2026-07-22T11:00:00Z").unwrap(),
            ),
            BTreeSet::from([
                "events/edge-m-codex/chunks/track.2026-07-22/new.jsonl".to_string()
            ])
        );
    }

    #[test]
    fn malformed_object_coverage_stays_unindexed() {
        let mut index = EventObjectIndex::empty("edge-m-codex", "2026-07-22");
        record_object(
            &mut index,
            "events/edge-m-codex/chunks/track.2026-07-22/unknown.jsonl",
            b"{\"event_id\":\"unknown\"}\n",
        );
        assert!(index.objects.is_empty());
    }

    #[test]
    fn legacy_days_are_unconditionally_queried_until_their_range_is_complete() {
        let mut index = EventPartitionIndex::empty("edge-m-codex");
        index.legacy_days.insert("2026-07-20".into());
        index.complete_legacy_day("2026-07-21", "2026-07-21T09:00:00Z", "2026-07-21T10:00:00Z");
        let since = parse_time("2026-07-22T00:00:00Z").unwrap();
        let until = parse_time("2026-07-23T00:00:00Z").unwrap();
        assert_eq!(
            index.candidate_days(since, until),
            BTreeSet::from(["2026-07-20".to_string()])
        );
    }

    #[test]
    fn initialization_marks_existing_physical_days_as_legacy() {
        let root =
            std::env::temp_dir().join(format!("synty-event-partitions-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bucket = crate::bucket::LocalFs::new(&root);
        bucket
            .put(
                "events/edge-m-codex/chunks/track.2026-07-21/one.jsonl",
                b"{}\n",
            )
            .unwrap();
        let index = load_or_initialize(&bucket, "edge-m-codex").unwrap();
        assert_eq!(
            index.legacy_days,
            BTreeSet::from(["2026-07-21".to_string()])
        );
        assert!(load(&bucket, "edge-m-codex").unwrap().is_some());
        let _ = std::fs::remove_dir_all(&root);
    }
}
