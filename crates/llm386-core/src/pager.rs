//! `Pager` trait — the seam between core and `llm386-pager`.

use thiserror::Error;

use crate::ids::BlockId;
use crate::page::{PagePlan, PageRequest};
use crate::retriever::RetrievalError;
use crate::store::StoreError;

pub trait Pager: Send + Sync {
    fn page(&self, request: PageRequest) -> Result<PagePlan, PagerError>;
}

#[derive(Debug, Error)]
pub enum PagerError {
    #[error("required block not found: {0}")]
    RequiredBlockMissing(BlockId),
    #[error("required blocks would exceed input budget")]
    RequiredOverBudget,
    #[error(transparent)]
    Retrieval(#[from] RetrievalError),
    #[error(transparent)]
    Storage(#[from] StoreError),
}
