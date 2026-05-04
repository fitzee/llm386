//! Adapters that wrap a Python object as a Rust trait impl.
//!
//! Lets users write a `Retriever` (and, in time, `Embedder` /
//! `Summarizer`) in Python and plug it into the same Rust pipeline
//! that runs the built-ins.

use std::sync::Arc;

use llm386_core::{BlockId, RetrievalCandidate, RetrievalError, Retriever, SessionId};
use pyo3::prelude::*;

/// Wraps a Python object that implements the retriever protocol:
///
/// ```python
/// class MyRetriever:
///     name = "my-retriever"
///
///     def retrieve(self, session: int, task: str, limit: int) -> list[tuple[str, float]]:
///         # Return a list of (block_id_hex, score) tuples.
///         ...
/// ```
///
/// The `name` attribute is read once at construction time and
/// `Box::leak`'d so the `&'static str` returned by `Retriever::name`
/// stays valid for the program's lifetime — registering many
/// adapters with distinct names is fine in practice (one small
/// leak per registration); reusing the same name across many
/// constructions would leak a fresh string each time.
pub struct PyRetriever {
    py_obj: Py<PyAny>,
    name: &'static str,
}

impl PyRetriever {
    pub fn new(py: Python<'_>, py_obj: Py<PyAny>) -> PyResult<Self> {
        let bound = py_obj.bind(py);
        let name: String = bound
            .getattr("name")
            .map_err(|e| {
                pyo3::exceptions::PyAttributeError::new_err(format!(
                    "python retriever object missing `name` attribute: {e}",
                ))
            })?
            .extract()
            .map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "python retriever `name` must be a str: {e}",
                ))
            })?;
        let name: &'static str = Box::leak(name.into_boxed_str());
        Ok(Self { py_obj, name })
    }
}

impl std::fmt::Debug for PyRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyRetriever").field("name", &self.name).finish_non_exhaustive()
    }
}

impl Retriever for PyRetriever {
    fn name(&self) -> &'static str {
        self.name
    }

    fn retrieve(
        &self,
        session: SessionId,
        task: &str,
        limit: usize,
    ) -> Result<Vec<RetrievalCandidate>, RetrievalError> {
        Python::attach(|py| {
            let bound = self.py_obj.bind(py);
            let result = bound
                .call_method1("retrieve", (session.0, task, limit))
                .map_err(|e| RetrievalError::Failed(format!("python retrieve: {e}")))?;
            let raw: Vec<(String, f32)> = result.extract().map_err(|e| {
                RetrievalError::Failed(format!(
                    "python retrieve must return list[tuple[str, float]]: {e}",
                ))
            })?;
            let mut cands = Vec::with_capacity(raw.len());
            for (id_str, score) in raw {
                let id = u128::from_str_radix(&id_str, 16).map_err(|e| {
                    RetrievalError::Failed(format!("invalid block id `{id_str}`: {e}"))
                })?;
                cands.push(RetrievalCandidate {
                    block_id: BlockId(id),
                    score: score.clamp(0.0, 1.0),
                    source: self.name.into(),
                });
            }
            Ok(cands)
        })
    }
}

/// Convenience constructor returning the wrapped Arc the pager
/// needs.
pub fn wrap_retriever(py: Python<'_>, py_obj: Py<PyAny>) -> PyResult<Arc<dyn Retriever>> {
    Ok(Arc::new(PyRetriever::new(py, py_obj)?))
}
