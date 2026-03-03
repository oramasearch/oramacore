use std::collections::HashMap;

use anyhow::{bail, Result};
use tracing::{info, warn};

use crate::types::{
    BoolFacetDefinition, DocumentId, FacetDefinition, FacetResult, FieldId, NumberFacetDefinition,
};

use super::{
    bool_field::BoolFieldStorage, number_field::NumberFieldStorage,
    path_to_index_id_map::PathToIndexId, string_filter_field::StringFilterFieldStorage, FieldType,
};

// =============================================================================
// Facetable trait with associated type
// =============================================================================

/// Trait for fields that support faceting operations.
///
/// This trait provides a uniform interface for calculating facets across
/// different field types (Bool, Number, StringFilter). Each implementation
/// specifies its own `FacetParam` type, enabling compile-time type safety.
///
/// # Type Parameters
///
/// - `FacetParam`: The type of facet definition this field accepts
///   (e.g., `BoolFacetDefinition`, `NumberFacetDefinition`, `StringFacetDefinition`)
///
/// # Design
///
/// The trait uses an associated type `FacetParam` to establish the relationship
/// between field types and their facet definition types. This allows the compiler
/// to enforce type safety and eliminates the need for runtime type matching.
///
/// Unlike the Filterable trait which returns document IDs, Facetable returns
/// the facet counts directly as a HashMap<String, usize> mapping facet values
/// to their counts.
pub trait Facetable {
    /// The type of facet definition this field accepts.
    ///
    /// Examples:
    /// - `BoolFacetDefinition` for Bool fields
    /// - `NumberFacetDefinition` for Number fields
    /// - `StringFacetDefinition` for StringFilter fields
    type FacetParam;

    /// Calculates facet counts based on the given facet definition.
    ///
    /// Returns a map from facet value labels to counts of matching documents.
    /// Only documents present in `token_scores` are counted.
    ///
    /// # Arguments
    ///
    /// * `facet_param` - The facet definition specifying what to count
    /// * `token_scores` - Map of document IDs to their scores; only these docs are counted
    ///
    /// # Returns
    ///
    /// A `Result` containing a HashMap mapping facet value labels (e.g., "true", "1-10")
    /// to counts, or an error if the faceting operation fails.
    fn calculate_facet(
        &self,
        facet_param: &Self::FacetParam,
        token_scores: &HashMap<DocumentId, f32>,
    ) -> Result<HashMap<String, usize>>;
}

// =============================================================================
// TryFromFacetDefinition trait
// =============================================================================

/// Trait for converting from the generic FacetDefinition enum to specific facet param types.
///
/// This trait enables type-safe conversion from the FacetDefinition enum variants to the
/// specific types expected by field implementations of the Facetable trait.
trait TryFromFacetDefinition {
    type Error;

    fn try_from_facet_definition(facet: &FacetDefinition) -> Result<&Self, Self::Error>
    where
        Self: Sized;
}

impl TryFromFacetDefinition for BoolFacetDefinition {
    type Error = anyhow::Error;

    fn try_from_facet_definition(facet: &FacetDefinition) -> Result<&Self, Self::Error> {
        if let FacetDefinition::Bool(bool_facet) = facet {
            Ok(bool_facet)
        } else {
            Err(anyhow::anyhow!(
                "Failed to convert facet definition to bool facet"
            ))
        }
    }
}

impl TryFromFacetDefinition for NumberFacetDefinition {
    type Error = anyhow::Error;

    fn try_from_facet_definition(facet: &FacetDefinition) -> Result<&Self, Self::Error> {
        if let FacetDefinition::Number(number_facet) = facet {
            Ok(number_facet)
        } else {
            Err(anyhow::anyhow!(
                "Failed to convert facet definition to number facet"
            ))
        }
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Calculates facet counts for a single field, combining uncommitted and committed results.
///
/// This generic function handles facet calculation for any field type that implements
/// the Facetable trait. It:
/// 1. Converts the generic FacetDefinition to the specific FacetParam type
/// 2. Calculates facets from the uncommitted field data
/// 3. Optionally calculates facets from the committed field data
/// 4. Combines results by summing counts for each facet value
///
/// # Type Parameters
/// * `UF` - Uncommitted field type implementing Facetable
/// * `CF` - Committed field type implementing Facetable with same FacetParam
/// * `FP` - Facet parameter type that can be converted from FacetDefinition
///
/// # Arguments
/// * `uncommitted_field` - The in-memory uncommitted field
/// * `committed_field` - Optional persisted committed field
/// * `facet_param` - The facet definition (generic FacetDefinition enum)
/// * `token_scores` - Map of matching document IDs to scores
///
/// # Returns
/// A HashMap mapping facet value labels to their counts
fn calculate_facet_on_field<UF, CF, FP>(
    uncommitted_field: Option<&UF>,
    committed_field: Option<&CF>,
    facet_param: &FacetDefinition,
    token_scores: &HashMap<DocumentId, f32>,
) -> Result<HashMap<String, usize>>
where
    UF: Facetable<FacetParam = FP>,
    CF: Facetable<FacetParam = FP>,
    FP: TryFromFacetDefinition,
{
    let facet_param: &FP = FP::try_from_facet_definition(facet_param)
        .map_err(|_| anyhow::anyhow!("Failed to convert facet definition to expected type"))?;

    let mut result = HashMap::new();

    // Process uncommitted field
    if let Some(uncommitted_field) = uncommitted_field {
        let uncommitted_facets = uncommitted_field.calculate_facet(facet_param, token_scores)?;
        for (label, count) in uncommitted_facets {
            *result.entry(label).or_default() += count;
        }
    }

    // Process committed field
    if let Some(committed_field) = committed_field {
        let committed_facets = committed_field.calculate_facet(facet_param, token_scores)?;
        for (label, count) in committed_facets {
            *result.entry(label).or_default() += count;
        }
    }

    Ok(result)
}

/// Routes facet operations to the appropriate field type.
///
/// This function acts as a dispatcher, matching the field type (Bool, Number,
/// StringFilter) and retrieving the corresponding field instances from
/// the uncommitted and committed field collections. It then delegates to
/// calculate_facet_on_field for the actual facet calculation.
///
/// # Arguments
/// * `field_id` - The ID of the field to calculate facets on
/// * `field_type` - The type of the field
/// * `uncommitted_fields` - Collection of all uncommitted fields
/// * `committed_fields` - Collection of all committed fields
/// * `facet_definition` - The facet criteria to apply
/// * `token_scores` - Map of matching document IDs to scores
///
/// # Returns
/// A HashMap mapping facet value labels to their counts, or an error if the field
/// type is not supported for faceting.
fn calculate_facet_for_field(
    field_id: FieldId,
    field_type: FieldType,
    bool_fields: &HashMap<FieldId, BoolFieldStorage>,
    number_fields: &HashMap<FieldId, NumberFieldStorage>,
    string_filter_fields: &HashMap<FieldId, StringFilterFieldStorage>,
    facet_definition: &FacetDefinition,
    token_scores: &HashMap<DocumentId, f32>,
) -> Result<HashMap<String, usize>> {
    match field_type {
        FieldType::Number => {
            let number_field = number_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Cannot find number field {field_id:?}"))?;

            let FacetDefinition::Number(facet_param) = facet_definition else {
                bail!("Expected Number facet definition for Number field");
            };

            number_field.calculate_facet(facet_param, token_scores)
        }
        FieldType::Bool => {
            let bool_field = bool_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Cannot find bool field {field_id:?}"))?;

            let FacetDefinition::Bool(facet_param) = facet_definition else {
                bail!("Expected Bool facet definition for Bool field");
            };

            Ok(bool_field.calculate_facet(facet_param, token_scores))
        }
        FieldType::StringFilter => {
            let string_filter_field = string_filter_fields
                .get(&field_id)
                .ok_or_else(|| anyhow::anyhow!("Cannot find string_filter field {field_id:?}"))?;

            let FacetDefinition::String(facet_param) = facet_definition else {
                bail!("Expected String facet definition for StringFilter field");
            };

            Ok(string_filter_field.calculate_facet(facet_param, token_scores))
        }
        _ => {
            bail!("Cannot calculate facet on field type {field_type:?}: unsupported for faceting");
        }
    }
}

// =============================================================================
// FacetParams and FacetContext
// =============================================================================

/// Parameters required for facet calculation.
///
/// This structure extracts only the necessary fields for faceting,
/// using lifetimes to avoid cloning data.
pub struct FacetParams<'search> {
    /// Map of field names to their facet definitions
    pub facets: &'search HashMap<String, FacetDefinition>,
    /// Map of document IDs to their search scores; only these documents are counted
    pub token_scores: &'search HashMap<DocumentId, f32>,
}

/// Context required for executing facet operations on an index.
///
/// This struct encapsulates all the data needed to calculate document facets,
/// allowing facet logic to be executed without direct Index coupling.
/// Index is responsible for gathering the data and passing it to the
/// FacetContext constructor, maintaining proper encapsulation.
pub struct FacetContext<'index> {
    path_to_index_id_map: &'index PathToIndexId,
    bool_fields: &'index HashMap<FieldId, BoolFieldStorage>,
    number_fields: &'index HashMap<FieldId, NumberFieldStorage>,
    string_filter_fields: &'index HashMap<FieldId, StringFilterFieldStorage>,
}

impl<'index> FacetContext<'index> {
    /// Creates a new FacetContext with the provided facet dependencies.
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

    /// Executes facet logic and populates the results map.
    ///
    /// This is the main entry point for faceting within a FacetContext.
    /// It applies the facet definitions and returns facet counts for each field.
    ///
    /// Note: This method is synchronous because all async operations (lock acquisition)
    /// are performed before creating the context.
    ///
    /// # Arguments
    /// * `params` - The facet parameters containing field definitions and token scores
    /// * `results` - Mutable map to populate with facet results
    ///
    /// # Returns
    /// * `Ok(())` - If faceting was calculated successfully
    /// * `Err(...)` - If faceting failed (e.g., unsupported field type)
    pub fn execute(
        self,
        params: &FacetParams<'_>,
        results: &mut HashMap<String, FacetResult>,
    ) -> Result<()> {
        if params.facets.is_empty() {
            return Ok(());
        }

        info!("Computing facets on {:?}", params.facets.keys());

        for (field_name, facet_definition) in params.facets {
            let Some((field_id, field_type)) =
                self.path_to_index_id_map.get_filter_field(field_name)
            else {
                warn!("Unknown field name '{}'", field_name);
                continue;
            };

            // Verify facet definition matches field type
            let is_valid_combination = matches!(
                (facet_definition, field_type),
                (FacetDefinition::Number(_), FieldType::Number)
                    | (FacetDefinition::Bool(_), FieldType::Bool)
                    | (FacetDefinition::String(_), FieldType::StringFilter)
            );

            if !is_valid_combination {
                bail!(
                    "Cannot calculate facet on field {field_name:?}: wrong type. Expected {field_type:?} facet"
                );
            }

            let facet_values = calculate_facet_for_field(
                field_id,
                field_type,
                self.bool_fields,
                self.number_fields,
                self.string_filter_fields,
                facet_definition,
                params.token_scores,
            )?;

            // Update or create facet result
            let facet_result = results
                .entry(field_name.clone())
                .or_insert_with(|| FacetResult {
                    count: 0,
                    values: HashMap::new(),
                });

            // Merge facet values
            for (label, count) in facet_values {
                *facet_result.values.entry(label).or_default() += count;
            }

            // Update total count of facet values
            facet_result.count = facet_result.values.len();
        }

        Ok(())
    }
}
