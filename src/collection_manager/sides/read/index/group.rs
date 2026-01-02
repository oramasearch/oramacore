use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use std::hash::Hash;

use crate::{
    collection_manager::sides::read::index::committed_field::{
        CommittedBoolField, CommittedNumberField, CommittedStringFilterField,
    },
    collection_manager::sides::read::index::uncommitted_field::{
        UncommittedBoolField, UncommittedNumberField, UncommittedStringFilterField,
    },
    types::{DocumentId, FieldId, Number, NumberFilter},
};

use super::{path_to_index_id_map::PathToIndexId, CommittedFields, FieldType, UncommittedFields};

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
// IntoGroupValue trait - converts specific types to GroupValue enum
// =============================================================================

/// Trait for converting specific group parameter types to GroupValue enum.
///
/// This is the reverse of TryFromFilter in filter.rs - we convert FROM specific
/// types TO the GroupValue enum. This enables type-safe grouping while still
/// allowing a unified return type.
trait IntoGroupValue {
    fn into_group_value(self) -> GroupValue;
}

impl IntoGroupValue for bool {
    fn into_group_value(self) -> GroupValue {
        GroupValue::Bool(self)
    }
}

impl IntoGroupValue for Number {
    fn into_group_value(self) -> GroupValue {
        GroupValue::Number(self)
    }
}

impl IntoGroupValue for String {
    fn into_group_value(self) -> GroupValue {
        GroupValue::String(self)
    }
}

// =============================================================================
// Groupable trait with associated type
// =============================================================================

/// Trait for fields that support grouping operations.
///
/// This trait provides a uniform interface for grouping documents across
/// different field types (Bool, Number, StringFilter). Each implementation
/// specifies its own `GroupParam` type, enabling compile-time type safety.
///
/// # Type Parameters
///
/// - `GroupParam`: The type of values this field produces for grouping
///   (e.g., `bool`, `Number`, `String`)
///
/// # Design
///
/// The trait uses an associated type `GroupParam` to establish the relationship
/// between field types and their group value types. This allows the compiler to
/// enforce type safety and eliminates the need for runtime type matching.
trait Groupable {
    /// The type of group parameter this field produces.
    ///
    /// Examples:
    /// - `bool` for Bool fields
    /// - `Number` for Number fields
    /// - `String` for StringFilter fields
    type GroupParam: IntoGroupValue;

    /// Returns an iterator over all distinct values in this field.
    fn get_values(&self) -> Box<dyn Iterator<Item = Self::GroupParam> + '_>;

    /// Returns an iterator over document IDs that have the specified value.
    ///
    /// # Arguments
    ///
    /// * `variant` - The specific value to filter by (type-safe, no enum matching)
    fn get_doc_ids(
        &self,
        variant: &Self::GroupParam,
    ) -> Result<Box<dyn Iterator<Item = DocumentId> + '_>>;
}

// =============================================================================
// Groupable implementations
// =============================================================================

impl Groupable for UncommittedBoolField {
    type GroupParam = bool;

    fn get_values(&self) -> Box<dyn Iterator<Item = bool> + '_> {
        Box::new(vec![true, false].into_iter())
    }

    fn get_doc_ids(&self, variant: &bool) -> Result<Box<dyn Iterator<Item = DocumentId> + '_>> {
        Ok(Box::new(self.filter(*variant)))
    }
}

impl Groupable for CommittedBoolField {
    type GroupParam = bool;

    fn get_values(&self) -> Box<dyn Iterator<Item = bool> + '_> {
        Box::new(vec![true, false].into_iter())
    }

    fn get_doc_ids(&self, variant: &bool) -> Result<Box<dyn Iterator<Item = DocumentId> + '_>> {
        Ok(Box::new(self.filter(*variant)?))
    }
}

impl Groupable for UncommittedNumberField {
    type GroupParam = Number;

    fn get_values(&self) -> Box<dyn Iterator<Item = Number> + '_> {
        Box::new(self.iter().map(|(n, _)| n))
    }

    fn get_doc_ids(&self, variant: &Number) -> Result<Box<dyn Iterator<Item = DocumentId> + '_>> {
        Ok(Box::new(self.filter(&NumberFilter::Equal(*variant))))
    }
}

impl Groupable for CommittedNumberField {
    type GroupParam = Number;

    fn get_values(&self) -> Box<dyn Iterator<Item = Number> + '_> {
        Box::new(self.iter().map(|(n, _)| n.0))
    }

    fn get_doc_ids(&self, variant: &Number) -> Result<Box<dyn Iterator<Item = DocumentId> + '_>> {
        Ok(Box::new(self.filter(&NumberFilter::Equal(*variant))?))
    }
}

impl Groupable for UncommittedStringFilterField {
    type GroupParam = String;

    fn get_values(&self) -> Box<dyn Iterator<Item = String> + '_> {
        Box::new(self.iter().map(|(n, _)| n))
    }

    fn get_doc_ids(&self, variant: &String) -> Result<Box<dyn Iterator<Item = DocumentId> + '_>> {
        Ok(Box::new(self.filter(variant)))
    }
}

impl Groupable for CommittedStringFilterField {
    type GroupParam = String;

    fn get_values(&self) -> Box<dyn Iterator<Item = String> + '_> {
        Box::new(self.iter().map(|(n, _)| n))
    }

    fn get_doc_ids(&self, variant: &String) -> Result<Box<dyn Iterator<Item = DocumentId> + '_>> {
        Ok(Box::new(self.filter(variant)))
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
    uncommitted_fields: &'index UncommittedFields,
    committed_fields: &'index CommittedFields,
}

impl<'index> GroupContext<'index> {
    /// Creates a new GroupContext with the provided group dependencies.
    ///
    /// This constructor accepts all necessary data as parameters, ensuring
    /// that Index internals are not directly accessed. The caller (Index)
    /// is responsible for gathering the data and passing it in.
    pub fn new(
        path_to_index_id_map: &'index PathToIndexId,
        uncommitted_fields: &'index UncommittedFields,
        committed_fields: &'index CommittedFields,
    ) -> Self {
        Self {
            path_to_index_id_map,
            uncommitted_fields,
            committed_fields,
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
                self.uncommitted_fields,
                self.committed_fields,
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

/// Maximum number of variants allowed for a single grouping field.
///
/// Grouping on fields with too many unique values would result in
/// an explosion of groups, which is likely not the intended behavior.
const MAX_GROUP_VARIANTS: usize = 500;

/// Calculates group values for a single field, combining uncommitted and committed data.
///
/// This generic function handles the combination of uncommitted (in-memory) and
/// committed (persisted) field data. It uses the `Groupable` trait with associated
/// types for compile-time type safety.
///
/// # Type Parameters
/// * `UF` - Uncommitted field type implementing Groupable
/// * `CF` - Committed field type implementing Groupable with same GroupParam
///
/// # Arguments
/// * `uncommitted_field` - The in-memory uncommitted field
/// * `committed_field` - Optional persisted committed field
///
/// # Returns
/// A map from GroupValue to the set of DocumentIds that have that value.
fn calculate_group_on_field<UF, CF>(
    uncommitted_field: &UF,
    committed_field: Option<&CF>,
) -> Result<HashMap<GroupValue, HashSet<DocumentId>>>
where
    UF: Groupable,
    CF: Groupable<GroupParam = UF::GroupParam>,
{
    let mut result = HashMap::new();

    // Process uncommitted field
    let variants: Vec<_> = uncommitted_field.get_values().collect();
    if variants.len() > MAX_GROUP_VARIANTS {
        bail!("Cannot calculate groups on a field with more than {MAX_GROUP_VARIANTS} variants");
    }

    for variant in variants {
        let docs: HashSet<DocumentId> = uncommitted_field.get_doc_ids(&variant)?.collect();
        let group_value = variant.into_group_value();
        let entry: &mut HashSet<DocumentId> = result.entry(group_value).or_default();
        entry.extend(docs);
    }

    // Process committed field (if present)
    if let Some(committed_field) = committed_field {
        let variants: Vec<_> = committed_field.get_values().collect();
        if variants.len() > MAX_GROUP_VARIANTS {
            bail!(
                "Cannot calculate groups on a field with more than {MAX_GROUP_VARIANTS} variants"
            );
        }

        for variant in variants {
            let docs: HashSet<DocumentId> = committed_field.get_doc_ids(&variant)?.collect();
            let group_value = variant.into_group_value();
            let entry: &mut HashSet<DocumentId> = result.entry(group_value).or_default();
            entry.extend(docs);
        }
    }

    Ok(result)
}

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
    uncommitted_fields: &UncommittedFields,
    committed_fields: &CommittedFields,
) -> Result<HashMap<GroupValue, HashSet<DocumentId>>> {
    match field_type {
        FieldType::Bool => {
            let uncommitted = uncommitted_fields
                .bool_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Cannot find bool field {field_id:?}"))?;
            let committed = committed_fields.bool_fields.get(&field_id);
            calculate_group_on_field(uncommitted, committed)
        }
        FieldType::Number => {
            let uncommitted = uncommitted_fields
                .number_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Cannot find number field {field_id:?}"))?;
            let committed = committed_fields.number_fields.get(&field_id);
            calculate_group_on_field(uncommitted, committed)
        }
        FieldType::StringFilter => {
            let uncommitted = uncommitted_fields
                .string_filter_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Cannot find string filter field {field_id:?}"))?;
            let committed = committed_fields.string_filter_fields.get(&field_id);
            calculate_group_on_field(uncommitted, committed)
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
