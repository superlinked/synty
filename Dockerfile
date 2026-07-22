FROM rust:1.96-bookworm AS builder

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked --features s3,gcs,mcp-http

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /var/synty synty \
    && mkdir -p /var/synty \
    && chown -R synty:synty /var/synty

COPY --from=builder /src/target/release/synty /usr/local/bin/synty

ENV SYNTY_HOME=/var/synty
USER 10001:10001
WORKDIR /var/synty
VOLUME ["/var/synty"]

ENTRYPOINT ["synty"]
CMD ["status", "--json"]
