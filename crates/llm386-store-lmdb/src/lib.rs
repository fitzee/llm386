//! `llm386-store-lmdb` — LMDB-backed persistent block storage.
//!
//! This crate provides [`LmdbStore`], an implementation of the
//! `BlockStore` trait from `llm386-core` backed by LMDB via the `heed`
//! binding. Blocks are content-hash deduplicated and indexed by both
//! id and session.

#![doc(html_root_url = "https://docs.rs/llm386-store-lmdb/1.0.0-alpha")]

mod store;

pub use store::{LmdbStore, StoreConfig, StoreOpenError};
