//! Fast replay parser for .replay26 protobuf files.
//!
//! Replaces the pure-Python protobuf decoder + state extraction with native Rust.
//! Exposes the same interface via PyO3 so it's a drop-in replacement.

use pyo3::prelude::*;

mod protobuf;
mod python;
mod replay;
mod sequential;
mod types;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
