//! PyO3 bindings for the LLM386 runtime.
//!
//! Exposes a `Store` class plus result types as a Python module.
//! The Python surface mirrors the v0 CLI-shelling SDK so code
//! written against either implementation works.

#![allow(unsafe_code)] // PyO3's macro expansion uses unsafe internally.

use pyo3::prelude::*;

mod store;
mod types;

use store::{LLM386Error, Store};
use types::{ChatMessage, ContextBlock, ModelProfile, OmittedBlock, PackResult, PagePlan, Provenance};

#[pymodule]
fn llm386(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Store>()?;
    m.add_class::<ContextBlock>()?;
    m.add_class::<Provenance>()?;
    m.add_class::<OmittedBlock>()?;
    m.add_class::<PagePlan>()?;
    m.add_class::<ChatMessage>()?;
    m.add_class::<PackResult>()?;
    m.add_class::<ModelProfile>()?;
    m.add("LLM386Error", py.get_type::<LLM386Error>())?;
    m.add_function(wrap_pyfunction!(list_models, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

/// Return every built-in model profile.
#[pyfunction]
fn list_models() -> Vec<ModelProfile> {
    llm386_core::default_registry()
        .profiles()
        .map(|p| ModelProfile::from_rust(p.clone()))
        .collect()
}
