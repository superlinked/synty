// pylate-rs ColBERT encoder. encode() returns a candle Tensor shaped
// [batch, max_len, dim] with zero-padding across the batch; doc_array() turns
// one document's rows into an Array2<f32>, dropping the all-zero pad rows.

use anyhow::{Context, Result};
use candle_core::Device;
use ndarray::Array2;
use pylate_rs::ColBERT;

pub struct Encoder {
    model: ColBERT,
}

impl Encoder {
    pub fn load(model_id: &str) -> Result<Self> {
        let dir = crate::model::ensure_model(model_id)?;
        let path = dir.to_str().ok_or_else(|| anyhow::anyhow!("non-utf8 model path"))?;
        let builder = ColBERT::from(path).with_device(select_device());
        let model = ColBERT::try_from(builder).map_err(|e| anyhow::anyhow!("load colbert: {e}"))?;
        Ok(Self { model })
    }

    pub fn encode_docs(&mut self, texts: &[String]) -> Result<Vec<Array2<f32>>> {
        self.encode(texts, false)
    }

    pub fn encode_query(&mut self, text: &str) -> Result<Array2<f32>> {
        self.encode(std::slice::from_ref(&text.to_string()), true)?
            .pop()
            .context("empty query encode")
    }

    fn encode(&mut self, texts: &[String], is_query: bool) -> Result<Vec<Array2<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        // One internal chunk per call (model batch_size=32) so per-chunk padded
        // tensors are never concatenated across chunks of differing length.
        for chunk in texts.chunks(16) {
            let t = self.model.encode(chunk, is_query).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            let (_b, _l, dim) = t.dims3().map_err(|e| anyhow::anyhow!("dims3: {e}"))?;
            let v3 = t.to_vec3::<f32>().map_err(|e| anyhow::anyhow!("to_vec3: {e}"))?;
            for doc in v3 {
                out.push(doc_array(doc, dim)?);
            }
        }
        Ok(out)
    }
}

/// Pick the encode device. With `--features metal` we try the Apple GPU and
/// fall back to CPU if it can't initialize; otherwise CPU (optionally with an
/// Accelerate/MKL BLAS backend compiled in). The choice is logged once.
fn select_device() -> Device {
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(d) => {
                eprintln!("encode: device = Metal (Apple GPU)");
                return d;
            }
            Err(e) => eprintln!("encode: Metal unavailable ({e}); using CPU"),
        }
    }
    #[cfg(all(not(feature = "metal"), any(feature = "accelerate", feature = "mkl")))]
    eprintln!("encode: device = CPU (BLAS backend enabled)");
    #[cfg(all(not(feature = "metal"), not(feature = "accelerate"), not(feature = "mkl")))]
    eprintln!("encode: device = CPU (plain; build with --features metal|accelerate for speed)");
    Device::Cpu
}

fn doc_array(doc: Vec<Vec<f32>>, dim: usize) -> Result<Array2<f32>> {
    let mut rows: Vec<Vec<f32>> = doc.into_iter().filter(|r| r.iter().any(|&x| x != 0.0)).collect();
    if rows.is_empty() {
        rows.push(vec![0.0; dim]);
    }
    let n = rows.len();
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n, dim), flat).context("array2 from shape")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Padding rows (all-zero) are dropped; real token rows survive in order.
    #[test]
    fn doc_array_drops_pad_rows() {
        let a = doc_array(vec![vec![1.0, 0.0], vec![0.0, 0.0], vec![2.0, 2.0]], 2).unwrap();
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.row(0).to_vec(), vec![1.0, 0.0]);
        assert_eq!(a.row(1).to_vec(), vec![2.0, 2.0]);
    }

    // A fully-masked document keeps a single zero row rather than an empty
    // matrix (next-plaid needs at least one row).
    #[test]
    fn doc_array_all_zero_keeps_one_row() {
        let a = doc_array(vec![vec![0.0, 0.0], vec![0.0, 0.0]], 2).unwrap();
        assert_eq!(a.shape(), &[1, 2]);
    }
}
