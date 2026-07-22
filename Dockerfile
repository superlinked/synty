# syntax=docker/dockerfile:1.7
FROM rust:1.97-bookworm AS builder

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
# Compile the locked dependency graph in a source-independent layer. A normal
# application edit then rebuilds only synty while registry/layer caches retain
# the expensive native dependencies on both architecture runners.
RUN mkdir src \
    && printf 'fn main() {}\n' > src/main.rs \
    && cargo build --release --locked --features s3,gcs,mcp-http \
    && rm -rf src
COPY src ./src
# Docker normalizes copied mtimes; force Cargo to invalidate the dummy package
# while retaining every dependency artifact from the preceding layer.
RUN touch src/main.rs \
    && cargo build --release --locked --features s3,gcs,mcp-http

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /var/synty synty \
    && mkdir -p /var/synty \
    && chown -R synty:synty /var/synty

COPY --from=builder /src/target/release/synty /usr/local/bin/synty

ENV SYNTY_HOME=/var/synty \
    HOME=/var/synty
USER 10001:10001
WORKDIR /var/synty
VOLUME ["/var/synty"]

ENTRYPOINT ["synty"]
CMD ["status", "--json"]
