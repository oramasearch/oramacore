use std::collections::HashSet;

use anyhow::{bail, Context, Result};
use oramacore_lib::filters::{FilterResult, PlainFilterResult};
use tracing::trace;

use crate::types::{DateFilter, DocumentId, Filter, GeoSearchFilter, NumberFilter, WhereFilter};

use super::{path_to_index_id_map::PathToIndexId, CommittedFields, FieldType, UncommittedFields};

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

/// Trait for converting from the generic Filter enum to specific filter parameter types.
///
/// This trait enables type-safe conversion from the Filter enum variants to the
/// specific types expected by field implementations of Filterable trait.
trait TryFromFilter {
    type Error;

    fn try_from_filter(filter: &Filter) -> Result<&Self, Self::Error>
    where
        Self: Sized;
}

impl TryFromFilter for bool {
    type Error = anyhow::Error;

    fn try_from_filter(filter: &Filter) -> Result<&Self, Self::Error> {
        if let Filter::Bool(b) = filter {
            Ok(b)
        } else {
            Err(anyhow::anyhow!("Failed to convert filter to bool"))
        }
    }
}

impl TryFromFilter for NumberFilter {
    type Error = anyhow::Error;

    fn try_from_filter(filter: &Filter) -> Result<&Self, Self::Error> {
        if let Filter::Number(num_filter) = filter {
            Ok(num_filter)
        } else {
            Err(anyhow::anyhow!("Failed to convert filter to number filter"))
        }
    }
}

impl TryFromFilter for DateFilter {
    type Error = anyhow::Error;

    fn try_from_filter(filter: &Filter) -> Result<&Self, Self::Error> {
        if let Filter::Date(date_filter) = filter {
            Ok(date_filter)
        } else {
            Err(anyhow::anyhow!("Failed to convert filter to date filter"))
        }
    }
}

impl TryFromFilter for String {
    type Error = anyhow::Error;

    fn try_from_filter(filter: &Filter) -> Result<&Self, Self::Error> {
        if let Filter::String(string_filter) = filter {
            Ok(string_filter)
        } else {
            Err(anyhow::anyhow!("Failed to convert filter to string"))
        }
    }
}

impl TryFromFilter for GeoSearchFilter {
    type Error = anyhow::Error;

    fn try_from_filter(filter: &Filter) -> Result<&Self, Self::Error> {
        if let Filter::GeoPoint(geo_search_filter) = filter {
            Ok(geo_search_filter)
        } else {
            Err(anyhow::anyhow!(
                "Failed to convert filter to geo search filter"
            ))
        }
    }
}

/// Filters a single field and combines uncommitted and committed results.
///
/// This generic function handles filtering for any field type that implements
/// the Filterable trait. It:
/// 1. Converts the generic Filter to the specific FilterParam type
/// 2. Filters the uncommitted field data
/// 3. Optionally filters the committed field data
/// 4. Combines results using OR logic (union of document sets)
///
/// # Type Parameters
/// * `UF` - Uncommitted field type implementing Filterable
/// * `CF` - Committed field type implementing Filterable
/// * `FP` - Filter parameter type that can be converted from Filter
///
/// # Arguments
/// * `document_count_estimate` - Total document count for result sizing
/// * `uncommitted_field` - The in-memory uncommitted field to filter
/// * `committed_field` - Optional persisted committed field to filter
/// * `filter_param` - The filter criteria (generic Filter enum)
///
/// # Returns
/// A FilterResult containing the union of matching document IDs from both fields
fn calculate_filter_on_field<UF, CF, FP>(
    document_count_estimate: u64,
    uncommitted_field: &UF,
    committed_field: Option<&CF>,
    filter_param: &Filter,
) -> Result<FilterResult<DocumentId>>
where
    UF: Filterable<FilterParam = FP>,
    CF: Filterable<FilterParam = FP>,
    FP: TryFromFilter,
{
    let filter_param: &FP = match FP::try_from_filter(filter_param) {
        Ok(p) => p,
        Err(_) => {
            // If the conversion fails, return empty set
            // This handles the following case:
            // - create a collection
            // - create an index1
            // - insert { "a": 10 } in index1
            // - create an index2
            // - insert { "a": "string" } in index2
            // - search with { where: { a: { gt: 5 } } }
            // In this case we consider only the index1
            // So, we return an empty set for index2
            // TODO: return a warning to the user
            return Ok(FilterResult::Filter(PlainFilterResult::new(
                document_count_estimate,
            )));
        }
    };

    let uncommitted_docs = uncommitted_field
        .filter(filter_param)
        .context("Failed to filter uncommitted field")?;

    let mut filtered = FilterResult::Filter(PlainFilterResult::from_iter(
        document_count_estimate,
        uncommitted_docs,
    ));

    // Combine uncommitted and committed results using OR logic
    if let Some(committed_field) = committed_field {
        let committed_docs = committed_field.filter(filter_param)?;
        filtered = FilterResult::or(
            filtered,
            FilterResult::Filter(PlainFilterResult::from_iter(
                document_count_estimate,
                committed_docs,
            )),
        );
    }

    Ok(filtered)
}

/// Routes filter operations to the appropriate field type.
///
/// This function acts as a dispatcher, matching the field type (Bool, Number, Date,
/// StringFilter, GeoPoint) and retrieving the corresponding field instances from
/// the uncommitted and committed field collections. It then delegates to
/// calculate_filter_on_field for the actual filtering.
///
/// # Arguments
/// * `document_count_estimate` - Total document count for result sizing
/// * `path_to_index_id_map` - Maps field paths to (FieldId, FieldType)
/// * `uncommitted_fields` - Collection of all uncommitted fields
/// * `committed_fields` - Collection of all committed fields
/// * `key` - The field path to filter on (e.g., "user.age")
/// * `filter` - The filter criteria to apply
///
/// # Returns
/// A FilterResult containing matching document IDs, or an error if the field
/// is not found or has an unsupported type.
fn calculate_filter_for_fields(
    document_count_estimate: u64,
    path_to_index_id_map: &PathToIndexId,
    uncommitted_fields: &UncommittedFields,
    committed_fields: &CommittedFields,
    key: &str,
    filter: &Filter,
) -> Result<FilterResult<DocumentId>> {
    let Some((field_id, field_type)) = path_to_index_id_map.get_filter_field(key) else {
        bail!("Field not found in index");
    };

    match field_type {
        FieldType::Bool => {
            let uncommitted_field = uncommitted_fields
                .bool_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Field not found in index"))?;

            calculate_filter_on_field(
                document_count_estimate,
                uncommitted_field,
                committed_fields.bool_fields.get(&field_id),
                filter,
            )
        }
        FieldType::Number => {
            let uncommitted_field = uncommitted_fields
                .number_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Field not found in index"))?;

            calculate_filter_on_field(
                document_count_estimate,
                uncommitted_field,
                committed_fields.number_fields.get(&field_id),
                filter,
            )
        }
        FieldType::Date => {
            let uncommitted_field = uncommitted_fields
                .date_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Field not found in index"))?;

            calculate_filter_on_field(
                document_count_estimate,
                uncommitted_field,
                committed_fields.date_fields.get(&field_id),
                filter,
            )
        }
        FieldType::StringFilter => {
            let uncommitted_field = uncommitted_fields
                .string_filter_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Field not found in index"))?;

            calculate_filter_on_field(
                document_count_estimate,
                uncommitted_field,
                committed_fields.string_filter_fields.get(&field_id),
                filter,
            )
        }
        FieldType::GeoPoint => {
            let uncommitted_field = uncommitted_fields
                .geopoint_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Field not found in index"))?;

            calculate_filter_on_field(
                document_count_estimate,
                uncommitted_field,
                committed_fields.geopoint_fields.get(&field_id),
                filter,
            )
        }
        _ => {
            // Unsupported field type - return empty set
            Ok(FilterResult::Filter(PlainFilterResult::new(
                document_count_estimate,
            )))
        }
    }
}

/// Calculates the filtered document IDs based on the provided where filter.
///
/// This is the main entry point for filter operations. It handles:
/// - Field-level filtering via filter_on_fields
/// - Recursive AND/OR/NOT combinations
/// - Validating fields exist in the index
///
/// The function processes a WhereFilter recursively, handling complex boolean
/// logic combinations. All field-level filters are combined with AND logic by
/// default, unless explicitly specified otherwise with OR/NOT operators.
///
/// # Arguments
/// * `document_count_estimate` - Total document count for result sizing
/// * `path_to_index_id_map` - Maps field paths to (FieldId, FieldType)
/// * `where_filter` - The filter criteria to apply
/// * `uncommitted_fields` - In-memory field data
/// * `committed_fields` - Persisted field data
///
/// # Returns
/// A FilterResult containing document IDs that match the filter criteria.
/// Returns an empty set if any specified field doesn't exist in the index.
fn calculate_filter(
    document_count_estimate: u64,
    path_to_index_id_map: &PathToIndexId,
    where_filter: &WhereFilter,
    uncommitted_fields: &UncommittedFields,
    committed_fields: &CommittedFields,
) -> Result<FilterResult<DocumentId>> {
    let mut results = Vec::new();

    for (k, filter) in &where_filter.filter_on_fields {
        // Check if field exists in the index
        if path_to_index_id_map.get_filter_field(k).is_none() {
            // If the user specified a field that is not in the index,
            // we should return an empty set.
            return Ok(FilterResult::Filter(PlainFilterResult::new(
                document_count_estimate,
            )));
        }

        // Use the new unified filtering approach with explicit type conversion
        let filtered = calculate_filter_for_fields(
            document_count_estimate,
            path_to_index_id_map,
            uncommitted_fields,
            committed_fields,
            k,
            filter,
        )?;

        results.push(filtered);
    }

    if let Some(filter) = where_filter.and.as_ref() {
        for f in filter {
            let result = calculate_filter(
                document_count_estimate,
                path_to_index_id_map,
                f,
                uncommitted_fields,
                committed_fields,
            )?;
            results.push(result);
        }
    }

    if let Some(filter) = where_filter.or.as_ref() {
        let mut or = vec![];
        for f in filter {
            let result = calculate_filter(
                document_count_estimate,
                path_to_index_id_map,
                f,
                uncommitted_fields,
                committed_fields,
            )?;
            or.push(result);
        }

        if or.is_empty() {
            return Ok(FilterResult::Filter(PlainFilterResult::new(
                document_count_estimate,
            )));
        }

        let mut result = or.pop().expect("we should have at least one result");
        for f in or {
            result = FilterResult::or(result, f);
        }

        results.push(result);
    }

    if let Some(filter) = where_filter.not.as_ref() {
        let result = calculate_filter(
            document_count_estimate,
            path_to_index_id_map,
            filter,
            uncommitted_fields,
            committed_fields,
        )?;
        results.push(FilterResult::Not(Box::new(result)));
    }

    // Is this correct????
    if results.is_empty() {
        return Ok(FilterResult::Filter(PlainFilterResult::new(
            document_count_estimate,
        )));
    }

    let mut result = results.pop().expect("we should have at least one result");
    for f in results {
        result = FilterResult::and(result, f);
    }

    Ok(result)
}

/// Context required for executing filter operations on an index.
///
/// This struct encapsulates all the data needed to filter documents,
/// allowing filter logic to be executed without direct Index coupling.
/// Index is responsible for gathering the data and passing it to the
/// FilterContext constructor, maintaining proper encapsulation.
pub struct FilterContext<'index> {
    document_count: u64,
    path_to_index_id_map: &'index PathToIndexId,
    uncommitted_fields: &'index UncommittedFields,
    committed_fields: &'index CommittedFields,
    uncommitted_deleted_documents: &'index HashSet<DocumentId>,
}

impl<'index> FilterContext<'index> {
    /// Creates a new FilterContext with the provided filter dependencies.
    ///
    /// This constructor accepts all necessary data as parameters, ensuring
    /// that Index internals are not directly accessed. The caller (Index)
    /// is responsible for gathering the data and passing it in.
    pub fn new(
        document_count: u64,
        path_to_index_id_map: &'index PathToIndexId,
        uncommitted_fields: &'index UncommittedFields,
        committed_fields: &'index CommittedFields,
        uncommitted_deleted_documents: &'index HashSet<DocumentId>,
    ) -> Self {
        Self {
            document_count,
            path_to_index_id_map,
            uncommitted_fields,
            committed_fields,
            uncommitted_deleted_documents,
        }
    }

    /// Executes filter logic and returns filtered document IDs.
    ///
    /// This is the main entry point for filtering within a FilterContext.
    /// It applies the WhereFilter criteria and integrates deleted documents.
    ///
    /// # Returns
    /// * `Ok(Some(FilterResult))` - If filter was applied successfully
    /// * `Ok(None)` - If the where_filter is empty (no filtering needed)
    /// * `Err(...)` - If filtering failed
    pub fn execute_filter(
        self,
        where_filter: &WhereFilter,
    ) -> Result<Option<FilterResult<DocumentId>>> {
        // Force an estimated size to avoid hash collisions in Bloom filters
        let expected_items = self.document_count.max(100_000);

        // Only return None if there's no filter AND no deleted documents
        if where_filter.is_empty() {
            if self.uncommitted_deleted_documents.is_empty() {
                return Ok(None);
            }
            // No filter but we have deleted documents - return NOT(deleted_docs)
            return Ok(Some(FilterResult::Not(Box::new(FilterResult::Filter(
                PlainFilterResult::from_iter(
                    expected_items,
                    self.uncommitted_deleted_documents.iter().copied(),
                ),
            )))));
        }

        trace!("Calculating filtered doc ids");

        let mut output = calculate_filter(
            expected_items,
            self.path_to_index_id_map,
            where_filter,
            self.uncommitted_fields,
            self.committed_fields,
        )?;

        // Integrate deleted documents using AND + NOT logic
        if !self.uncommitted_deleted_documents.is_empty() {
            output = FilterResult::and(
                output,
                FilterResult::Not(Box::new(FilterResult::Filter(
                    PlainFilterResult::from_iter(
                        expected_items,
                        self.uncommitted_deleted_documents.iter().copied(),
                    ),
                ))),
            );
        }

        Ok(Some(output))
    }
}
