use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};
use oramacore_lib::filters::{FilterResult, PlainFilterResult};
use tracing::trace;

use crate::types::{DocumentId, FieldId, Filter, WhereFilter};

use super::{
    bool_field::BoolFieldStorage, date_field::DateFieldStorage,
    geopoint_field::GeoPointFieldStorage, number_field::NumberFieldStorage,
    path_to_index_id_map::PathToIndexId, string_filter_field::StringFilterFieldStorage, FieldType,
};

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
#[allow(clippy::too_many_arguments)]
fn calculate_filter_for_fields(
    document_count_estimate: u64,
    path_to_index_id_map: &PathToIndexId,
    bool_fields: &HashMap<FieldId, BoolFieldStorage>,
    number_fields: &HashMap<FieldId, NumberFieldStorage>,
    date_fields: &HashMap<FieldId, DateFieldStorage>,
    geopoint_fields: &HashMap<FieldId, GeoPointFieldStorage>,
    string_filter_fields: &HashMap<FieldId, StringFilterFieldStorage>,
    key: &str,
    filter: &Filter,
) -> Result<FilterResult<DocumentId>> {
    let Some((field_id, field_type)) = path_to_index_id_map.get_filter_field(key) else {
        bail!("Field not found in index");
    };

    match field_type {
        FieldType::Bool => {
            let bool_field = bool_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Bool field not found in index"))?;

            let Filter::Bool(bool_value) = filter else {
                // Wrong filter type for bool field - return empty set
                return Ok(FilterResult::Filter(PlainFilterResult::new(
                    document_count_estimate,
                )));
            };

            let docs = bool_field
                .filter_docs(*bool_value)
                .into_iter()
                .map(DocumentId);
            Ok(FilterResult::Filter(PlainFilterResult::from_iter(
                document_count_estimate,
                docs,
            )))
        }
        FieldType::Number => {
            let number_field = number_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Number field not found in index"))?;

            let Filter::Number(number_filter) = filter else {
                // Wrong filter type for number field - return empty set
                return Ok(FilterResult::Filter(PlainFilterResult::new(
                    document_count_estimate,
                )));
            };

            let docs = number_field.filter(number_filter);
            Ok(FilterResult::Filter(PlainFilterResult::from_iter(
                document_count_estimate,
                docs,
            )))
        }
        FieldType::Date => {
            let date_field = date_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Date field not found in index"))?;

            let Filter::Date(date_filter) = filter else {
                // Wrong filter type for date field - return empty set
                return Ok(FilterResult::Filter(PlainFilterResult::new(
                    document_count_estimate,
                )));
            };

            let docs = date_field.filter(date_filter);
            Ok(FilterResult::Filter(PlainFilterResult::from_iter(
                document_count_estimate,
                docs.into_iter(),
            )))
        }
        FieldType::StringFilter => {
            let string_filter_field = string_filter_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("StringFilter field not found in index"))?;

            let Filter::String(string_value) = filter else {
                // Wrong filter type for string_filter field - return empty set
                return Ok(FilterResult::Filter(PlainFilterResult::new(
                    document_count_estimate,
                )));
            };

            let docs = string_filter_field.filter_docs(string_value);
            Ok(FilterResult::Filter(PlainFilterResult::from_iter(
                document_count_estimate,
                docs.into_iter(),
            )))
        }
        FieldType::GeoPoint => {
            let geopoint_field = geopoint_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("GeoPoint field not found in index"))?;

            let Filter::GeoPoint(geo_filter) = filter else {
                // Wrong filter type for geopoint field - return empty set
                return Ok(FilterResult::Filter(PlainFilterResult::new(
                    document_count_estimate,
                )));
            };

            let docs = geopoint_field
                .filter(geo_filter)
                .context("Failed to filter geopoint field")?;
            Ok(FilterResult::Filter(PlainFilterResult::from_iter(
                document_count_estimate,
                docs.into_iter(),
            )))
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
#[allow(clippy::too_many_arguments)]
fn calculate_filter(
    document_count_estimate: u64,
    path_to_index_id_map: &PathToIndexId,
    bool_fields: &HashMap<FieldId, BoolFieldStorage>,
    number_fields: &HashMap<FieldId, NumberFieldStorage>,
    date_fields: &HashMap<FieldId, DateFieldStorage>,
    geopoint_fields: &HashMap<FieldId, GeoPointFieldStorage>,
    string_filter_fields: &HashMap<FieldId, StringFilterFieldStorage>,
    where_filter: &WhereFilter,
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
            bool_fields,
            number_fields,
            date_fields,
            geopoint_fields,
            string_filter_fields,
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
                bool_fields,
                number_fields,
                date_fields,
                geopoint_fields,
                string_filter_fields,
                f,
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
                bool_fields,
                number_fields,
                date_fields,
                geopoint_fields,
                string_filter_fields,
                f,
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
            bool_fields,
            number_fields,
            date_fields,
            geopoint_fields,
            string_filter_fields,
            filter,
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
    bool_fields: &'index HashMap<FieldId, BoolFieldStorage>,
    number_fields: &'index HashMap<FieldId, NumberFieldStorage>,
    date_fields: &'index HashMap<FieldId, DateFieldStorage>,
    geopoint_fields: &'index HashMap<FieldId, GeoPointFieldStorage>,
    string_filter_fields: &'index HashMap<FieldId, StringFilterFieldStorage>,
    uncommitted_deleted_documents: &'index HashSet<DocumentId>,
}

impl<'index> FilterContext<'index> {
    /// Creates a new FilterContext with the provided filter dependencies.
    ///
    /// This constructor accepts all necessary data as parameters, ensuring
    /// that Index internals are not directly accessed. The caller (Index)
    /// is responsible for gathering the data and passing it in.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        document_count: u64,
        path_to_index_id_map: &'index PathToIndexId,
        bool_fields: &'index HashMap<FieldId, BoolFieldStorage>,
        number_fields: &'index HashMap<FieldId, NumberFieldStorage>,
        date_fields: &'index HashMap<FieldId, DateFieldStorage>,
        geopoint_fields: &'index HashMap<FieldId, GeoPointFieldStorage>,
        string_filter_fields: &'index HashMap<FieldId, StringFilterFieldStorage>,
        uncommitted_deleted_documents: &'index HashSet<DocumentId>,
    ) -> Self {
        Self {
            document_count,
            path_to_index_id_map,
            bool_fields,
            number_fields,
            date_fields,
            geopoint_fields,
            string_filter_fields,
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
            self.bool_fields,
            self.number_fields,
            self.date_fields,
            self.geopoint_fields,
            self.string_filter_fields,
            where_filter,
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
