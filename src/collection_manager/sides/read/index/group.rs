use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use std::hash::Hash;

use crate::types::{DocumentId, FieldId, Number};

use super::{
    bool_field::BoolFieldStorage, number_field::NumberFieldStorage,
    path_to_index_id_map::PathToIndexId, string_filter_field::StringFilterFieldStorage, FieldType,
};

// =============================================================================
// GroupValue enum
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupValue {
    String(String),
    Number(Number),
    Bool(bool),
}

impl Hash for GroupValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            GroupValue::String(string) => string.hash(state),
            GroupValue::Number(Number::I32(number)) => number.hash(state),
            GroupValue::Number(Number::F32(number)) => {
                f32::to_be_bytes(*number).hash(state);
            }
            GroupValue::Bool(bool) => bool.hash(state),
        }
    }
}

impl From<GroupValue> for serde_json::Value {
    fn from(val: GroupValue) -> Self {
        match val {
            GroupValue::String(string) => serde_json::Value::String(string),
            GroupValue::Number(number) => number.into(),
            GroupValue::Bool(b) => serde_json::Value::Bool(b),
        }
    }
}

// =============================================================================
// GroupParams and GroupContext
// =============================================================================

/// Parameters required for group calculation.
///
/// This structure extracts only the necessary fields for grouping,
/// using lifetimes to avoid cloning data.
pub struct GroupParams<'search> {
    /// Field names to group by
    pub properties: &'search [String],
}

/// Context required for executing group operations on an index.
///
/// This struct encapsulates all the data needed to calculate document groups,
/// allowing group logic to be executed without direct Index coupling.
/// Index is responsible for gathering the data and passing it to the
/// GroupContext constructor, maintaining proper encapsulation.
pub struct GroupContext<'index> {
    path_to_index_id_map: &'index PathToIndexId,
    bool_fields: &'index HashMap<FieldId, BoolFieldStorage>,
    number_fields: &'index HashMap<FieldId, NumberFieldStorage>,
    string_filter_fields: &'index HashMap<FieldId, StringFilterFieldStorage>,
}

impl<'index> GroupContext<'index> {
    /// Creates a new GroupContext with the provided group dependencies.
    ///
    /// This constructor accepts all necessary data as parameters, ensuring
    /// that Index internals are not directly accessed. The caller (Index)
    /// is responsible for gathering the data and passing it in.
    pub fn new(
        path_to_index_id_map: &'index PathToIndexId,
        bool_fields: &'index HashMap<FieldId, BoolFieldStorage>,
        number_fields: &'index HashMap<FieldId, NumberFieldStorage>,
        string_filter_fields: &'index HashMap<FieldId, StringFilterFieldStorage>,
    ) -> Self {
        Self {
            path_to_index_id_map,
            bool_fields,
            number_fields,
            string_filter_fields,
        }
    }

    /// Executes group logic and populates the results map.
    ///
    /// This is the main entry point for grouping within a GroupContext.
    /// It applies the grouping criteria and returns document ID sets for each group.
    ///
    /// Note: This method is synchronous because all async operations (lock acquisition)
    /// are performed before creating the context.
    ///
    /// # Returns
    /// * `Ok(())` - If grouping was calculated successfully
    /// * `Err(...)` - If grouping failed (e.g., field not found, too many variants)
    pub fn execute(
        self,
        params: &GroupParams<'_>,
        results: &mut HashMap<Vec<GroupValue>, HashSet<DocumentId>>,
    ) -> Result<()> {
        // Resolve field names to (FieldId, FieldType) pairs
        // All properties must exist in this index, otherwise skip this index entirely
        let Some(properties) = params
            .properties
            .iter()
            .map(|field_name| self.path_to_index_id_map.get_filter_field(field_name))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(());
        };

        // Calculate all variants and their document sets for each field
        let mut all_variants = HashMap::<FieldId, HashMap<GroupValue, HashSet<DocumentId>>>::new();
        for (field_id, field_type) in &properties {
            let field_variants = calculate_group_for_field(
                *field_id,
                *field_type,
                self.bool_fields,
                self.number_fields,
                self.string_filter_fields,
            )?;
            all_variants.insert(*field_id, field_variants);
        }

        // Generate all combinations of group values
        let field_ids: Vec<_> = properties.iter().map(|(field_id, _)| *field_id).collect();
        let mut groups = Vec::new();
        generate_group_combinations(&field_ids, &all_variants, &mut Vec::new(), &mut groups);

        // Calculate document intersections for each combination
        for group_combination in groups {
            // Find intersection of all document sets for this combination
            let mut intersection: Option<HashSet<DocumentId>> = None;
            let mut group_values = Vec::new();

            for (field_id, group_value) in group_combination {
                let Some(docs) = all_variants
                    .get(&field_id)
                    .and_then(|map| map.get(&group_value))
                else {
                    continue;
                };

                intersection = if let Some(current_intersection) = intersection {
                    Some(current_intersection.intersection(docs).cloned().collect())
                } else {
                    Some(docs.clone())
                };

                group_values.push(group_value);
            }

            let doc_ids_for_key = results.entry(group_values).or_default();
            if let Some(intersection) = intersection {
                doc_ids_for_key.extend(intersection);
            }
        }

        Ok(())
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Routes group operations to the appropriate field type.
///
/// This function acts as a dispatcher, matching the field type (Bool, Number,
/// StringFilter) and calling `calculate_group_on_field` with the concrete
/// field types. This approach provides full type safety since each branch
/// works with concrete types rather than trait objects.
///
/// # Arguments
/// * `field_id` - The ID of the field to group on
/// * `field_type` - The type of the field
/// * `uncommitted_fields` - The in-memory uncommitted field collection
/// * `committed_fields` - The persisted committed field collection
///
/// # Returns
/// A map from GroupValue to the set of DocumentIds that have that value.
fn calculate_group_for_field(
    field_id: FieldId,
    field_type: FieldType,
    bool_fields: &HashMap<FieldId, BoolFieldStorage>,
    number_fields: &HashMap<FieldId, NumberFieldStorage>,
    string_filter_fields: &HashMap<FieldId, StringFilterFieldStorage>,
) -> Result<HashMap<GroupValue, HashSet<DocumentId>>> {
    match field_type {
        FieldType::Bool => {
            let bool_field = bool_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Cannot find bool field {field_id:?}"))?;

            // Use BoolFieldStorage::get_grouped_docs() which returns HashMap<bool, HashSet<DocumentId>>
            let grouped = bool_field.get_grouped_docs();
            let mut result = HashMap::new();
            for (bool_val, doc_ids) in grouped {
                result.insert(GroupValue::Bool(bool_val), doc_ids);
            }
            Ok(result)
        }
        FieldType::Number => {
            let number_field = number_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Cannot find number field {field_id:?}"))?;

            Ok(number_field.get_grouped_docs())
        }
        FieldType::StringFilter => {
            let string_filter_field = string_filter_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Cannot find string filter field {field_id:?}"))?;

            // Use StringFilterFieldStorage::get_grouped_docs() directly
            let grouped = string_filter_field.get_grouped_docs();
            let mut result = HashMap::with_capacity(grouped.len());
            for (key, doc_ids) in grouped {
                result.insert(GroupValue::String(key), doc_ids);
            }
            Ok(result)
        }
        FieldType::GeoPoint => {
            bail!("Cannot calculate group on a GeoPoint field")
        }
        _ => {
            bail!("Cannot calculate group on {field_type:?} field")
        }
    }
}

/// Recursively generates all combinations of group values for multiple properties.
///
/// Creates the Cartesian product of all group variants from each field.
/// For example, if field1 has values [A, B] and field2 has values [X, Y],
/// this generates: [(A, X), (A, Y), (B, X), (B, Y)].
///
/// # Arguments
/// * `field_ids` - Ordered list of field IDs to generate combinations for
/// * `all_variants` - Map from field ID to its variant->documents mapping
/// * `current_combination` - Working buffer for the current combination being built
/// * `result` - Output vector to collect all generated combinations
fn generate_group_combinations(
    field_ids: &[FieldId],
    all_variants: &HashMap<FieldId, HashMap<GroupValue, HashSet<DocumentId>>>,
    current_combination: &mut Vec<(FieldId, GroupValue)>,
    result: &mut Vec<Vec<(FieldId, GroupValue)>>,
) {
    if current_combination.len() == field_ids.len() {
        result.push(current_combination.clone());
        return;
    }

    let field_id = field_ids[current_combination.len()];
    if let Some(variants) = all_variants.get(&field_id) {
        for group_value in variants.keys() {
            current_combination.push((field_id, group_value.clone()));
            generate_group_combinations(field_ids, all_variants, current_combination, result);
            current_combination.pop();
        }
    }
}
