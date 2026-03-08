mod tokenizer;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use tokenizer::Tokenizer;

#[pymodule]
fn fast_cs336(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}
