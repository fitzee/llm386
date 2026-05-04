//! Crate-level error aggregator.

use thiserror::Error;

use crate::packer::PackerError;
use crate::pager::PagerError;
use crate::retriever::RetrievalError;
use crate::store::StoreError;
use crate::tokenizer::TokenizerError;
use crate::trace::TraceError;

/// Top-level error type aggregating every crate-specific error.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    Retrieval(#[from] RetrievalError),
    #[error(transparent)]
    Pager(#[from] PagerError),
    #[error(transparent)]
    Packer(#[from] PackerError),
    #[error(transparent)]
    Trace(#[from] TraceError),
}
