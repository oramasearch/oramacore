use anyhow::Result;

use crate::types::DocumentId;

/// Trait for fields that support filtering operations.
///
/// This trait provides a uniform interface for filtering documents across
/// different field types (Bool, Number, Date, StringFilter, GeoPoint) with their
/// corresponding filter parameter types.
///
/// # Type Parameters
///
/// - `FilterParam`: The type of filter criteria this field accepts
///   (e.g., `bool`, `NumberFilter`, `DateFilter`, `String`, `GeoSearchFilter`)
///
/// # Design
///
/// The trait uses an associated type `FilterParam` to establish the relationship
/// between field types and their filter parameters. This allows the compiler to
/// enforce type safety while enabling generic filter operations.
///
/// The trait normalizes different filter method signatures across field types:
/// - Some fields return `Result<impl Iterator>` (Bool, Number, Date)
/// - Some fields return `impl Iterator` without Result (StringFilter)
/// - Some fields return `Box<dyn Iterator>` (GeoPoint)
///
/// By returning `Result<Box<dyn Iterator>>`, the trait provides a uniform interface.
pub trait Filterable {
    /// The type of filter parameter this field accepts.
    ///
    /// Examples:
    /// - `bool` for Bool fields
    /// - `NumberFilter` for Number fields
    /// - `DateFilter` for Date fields
    /// - `String` for StringFilter fields
    /// - `GeoSearchFilter` for GeoPoint fields
    type FilterParam;

    /// Filters documents based on the given filter parameter.
    ///
    /// Returns an iterator of `DocumentId`s that match the filter criteria.
    /// The iterator yields documents from this field that satisfy the filter.
    ///
    /// # Arguments
    ///
    /// * `filter_param` - The filter criteria to apply
    ///
    /// # Returns
    ///
    /// A `Result` containing a boxed iterator of matching `DocumentId`s, or an error
    /// if the filtering operation fails (e.g., due to index corruption).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The underlying index data is corrupted
    /// - I/O errors occur when reading committed data from disk
    fn filter<'s, 'iter>(
        &'s self,
        filter_param: &Self::FilterParam,
    ) -> Result<Box<dyn Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter;
}
