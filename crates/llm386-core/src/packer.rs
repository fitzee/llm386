//! `Packer` trait — the seam between core and `llm386-packer`.

use thiserror::Error;

use crate::model::ModelProfile;
use crate::packed::PackedPrompt;
use crate::page::PagePlan;
use crate::store::StoreError;
use crate::tokenizer::TokenizerError;

pub trait Packer: Send + Sync {
    fn pack(&self, plan: &PagePlan, model: &ModelProfile) -> Result<PackedPrompt, PackerError>;
}

#[derive(Debug, Error)]
pub enum PackerError {
    #[error("packed prompt would exceed input budget ({tokens} > {budget})")]
    BudgetExceeded { tokens: u32, budget: u32 },
    #[error(transparent)]
    Storage(#[from] StoreError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
}
