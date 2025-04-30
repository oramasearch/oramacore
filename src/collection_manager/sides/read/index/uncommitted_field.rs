pub use bool::{UncommittedBoolField, UncommittedBoolFieldStats};
pub use number::{UncommittedNumberField, UncommittedNumberFieldStats};
pub use string::{UncommittedStringField, UncommittedStringFieldStats};
pub use string_filter::{UncommittedStringFilterField, UncommittedStringFilterFieldStats};
pub use vector::{UncommittedVectorField, UncommittedVectorFieldStats};

mod bool;
mod number;
mod string;
mod string_filter;
mod vector;

pub use string::{Positions, TotalDocumentsWithTermInField};
