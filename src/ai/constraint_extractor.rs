//! Schema-aware constraint extraction from natural language queries.
//!
//! This module extracts numeric, string enum, and boolean constraints from user queries
//! using regex patterns and schema information. It also validates LLM-generated search
//! params and injects any constraints the LLM missed.

use regex::Regex;
use std::collections::HashMap;

use crate::types::{Filter, NumberFilter, SearchParams, WhereFilter};

// ===== Core Types =====

/// A constraint extracted from the user's natural language query.
#[derive(Debug, Clone)]
pub enum ExtractedConstraint {
    /// Numeric constraint like "under $100", "at least 4 stars", "between 50 and 150"
    Numeric {
        original_text: String,
        operator: NumericOp,
        value: f64,
        /// Upper bound for Between operator
        upper: Option<f64>,
        /// Hint for which field this constraint applies to (e.g., "price", "rating")
        field_hint: Option<String>,
    },
    /// String enum constraint matched against known schema values
    StringEnum {
        original_text: String,
        /// The exact enum value from the schema that was matched
        matched_value: String,
        /// The field this constraint belongs to
        field_name: String,
    },
    /// Boolean constraint like "in stock" -> true, "out of stock" -> false
    Boolean {
        original_text: String,
        value: bool,
        /// Hint for which field this constraint applies to
        field_hint: Option<String>,
    },
}

/// Numeric comparison operators
#[derive(Debug, Clone, PartialEq)]
pub enum NumericOp {
    Lte,
    Gte,
    Eq,
    Between,
}

/// Information about a field in the schema
#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub field_type: String,
}

/// Budget allocation for a sub-query
#[derive(Debug, Clone, serde::Deserialize)]
pub struct BudgetAllocation {
    pub query_index: usize,
    pub budget_cap: f64,
    pub field: String,
}

/// Budget planner response from the LLM
#[derive(Debug, Clone, serde::Deserialize)]
pub struct BudgetPlannerResponse {
    pub allocations: Vec<BudgetAllocation>,
    #[allow(dead_code)]
    pub strategy: String,
}

// ===== Extraction =====

/// Extract all constraints from a natural language query using the schema context.
///
/// Uses regex patterns for numeric and boolean extraction, and fuzzy-matches
/// string enum values from the `filter_properties` map.
pub fn extract_constraints(
    query: &str,
    schema_fields: &[FieldInfo],
    filter_properties: &HashMap<String, Vec<String>>,
) -> Vec<ExtractedConstraint> {
    let mut constraints = Vec::new();

    constraints.extend(extract_numeric_constraints(query));
    constraints.extend(extract_string_enum_constraints(query, filter_properties));
    constraints.extend(extract_boolean_constraints(query, schema_fields));

    constraints
}

/// Extract numeric constraints using regex patterns.
///
/// Recognizes patterns like:
/// - "under/below/less than/max/up to $X" -> Lte
/// - "over/above/more than/min/at least $X" -> Gte
/// - "between $X and $Y" / "$X-$Y" -> Between
/// - "exactly $X" -> Eq
fn extract_numeric_constraints(query: &str) -> Vec<ExtractedConstraint> {
    let mut constraints = Vec::new();
    let lower = query.to_lowercase();

    // Currency/unit prefix pattern: matches $, USD, EUR, GBP, etc.
    // Used in all numeric regexes to handle various currency notations.
    let cur = r"(?:\$|usd|eur|gbp|£|€)?\s*";

    // Pattern: "between X and Y" or "from X to Y"
    let between_re = Regex::new(&format!(
        r"(?i)(?:between|from)\s+{cur}(\d+(?:\.\d+)?)\s+(?:and|to)\s+{cur}(\d+(?:\.\d+)?)"
    ))
    .expect("Valid regex");

    for cap in between_re.captures_iter(&lower) {
        if let (Ok(low), Ok(high)) = (cap[1].parse::<f64>(), cap[2].parse::<f64>()) {
            let original = cap[0].to_string();
            let hint =
                guess_field_hint_from_context(&lower, cap.get(0).map(|m| m.start()).unwrap_or(0));
            constraints.push(ExtractedConstraint::Numeric {
                original_text: original,
                operator: NumericOp::Between,
                value: low,
                upper: Some(high),
                field_hint: hint,
            });
        }
    }

    // Pattern: "$X-$Y" range notation (e.g., "$50-$150", "USD50-USD150")
    // Requires at least one currency prefix to avoid matching arbitrary number ranges
    let range_re = Regex::new(
        r"(?i)(?:\$|usd|eur|gbp|£|€)\s*(\d+(?:\.\d+)?)\s*[-–]\s*(?:\$|usd|eur|gbp|£|€)?\s*(\d+(?:\.\d+)?)",
    )
    .expect("Valid regex");

    for cap in range_re.captures_iter(&lower) {
        if let (Ok(low), Ok(high)) = (cap[1].parse::<f64>(), cap[2].parse::<f64>()) {
            // Skip if already captured by between pattern
            let already_captured = constraints.iter().any(|c| {
                if let ExtractedConstraint::Numeric {
                    value,
                    upper: Some(u),
                    ..
                } = c
                {
                    (*value - low).abs() < 0.01 && (*u - high).abs() < 0.01
                } else {
                    false
                }
            });
            if !already_captured {
                let hint = guess_field_hint_from_context(
                    &lower,
                    cap.get(0).map(|m| m.start()).unwrap_or(0),
                );
                constraints.push(ExtractedConstraint::Numeric {
                    original_text: cap[0].to_string(),
                    operator: NumericOp::Between,
                    value: low,
                    upper: Some(high),
                    field_hint: hint,
                });
            }
        }
    }

    // Pattern: "under/below/less than/max/up to/no more than/within/budget of" + number -> Lte
    let lte_re = Regex::new(&format!(
        r"(?i)(?:under|below|less\s+than|max(?:imum)?|up\s+to|no\s+more\s+than|within|budget\s+(?:of\s+)?|cheaper\s+than)\s*{cur}(\d+(?:\.\d+)?)"
    )).expect("Valid regex");

    for cap in lte_re.captures_iter(&lower) {
        if let Ok(val) = cap[1].parse::<f64>() {
            // Skip if already part of a between constraint
            let already_captured = constraints.iter().any(|c| {
                if let ExtractedConstraint::Numeric {
                    operator: NumericOp::Between,
                    ..
                } = c
                {
                    true
                } else {
                    false
                }
            });
            if !already_captured {
                let hint = guess_field_hint_from_context(
                    &lower,
                    cap.get(0).map(|m| m.start()).unwrap_or(0),
                );
                constraints.push(ExtractedConstraint::Numeric {
                    original_text: cap[0].to_string(),
                    operator: NumericOp::Lte,
                    value: val,
                    upper: None,
                    field_hint: hint,
                });
            }
        }
    }

    // Pattern: "over/above/more than/min/at least/starting from" + number -> Gte
    let gte_re = Regex::new(&format!(
        r"(?i)(?:over|above|more\s+than|min(?:imum)?|at\s+least|starting\s+from|no\s+less\s+than)\s*{cur}(\d+(?:\.\d+)?)"
    )).expect("Valid regex");

    for cap in gte_re.captures_iter(&lower) {
        if let Ok(val) = cap[1].parse::<f64>() {
            let already_captured = constraints.iter().any(|c| {
                if let ExtractedConstraint::Numeric {
                    operator: NumericOp::Between,
                    ..
                } = c
                {
                    true
                } else {
                    false
                }
            });
            if !already_captured {
                let hint = guess_field_hint_from_context(
                    &lower,
                    cap.get(0).map(|m| m.start()).unwrap_or(0),
                );
                constraints.push(ExtractedConstraint::Numeric {
                    original_text: cap[0].to_string(),
                    operator: NumericOp::Gte,
                    value: val,
                    upper: None,
                    field_hint: hint,
                });
            }
        }
    }

    // Pattern: "exactly" + number -> Eq
    let eq_re = Regex::new(&format!(r"(?i)exactly\s+{cur}(\d+(?:\.\d+)?)")).expect("Valid regex");

    for cap in eq_re.captures_iter(&lower) {
        if let Ok(val) = cap[1].parse::<f64>() {
            let hint =
                guess_field_hint_from_context(&lower, cap.get(0).map(|m| m.start()).unwrap_or(0));
            constraints.push(ExtractedConstraint::Numeric {
                original_text: cap[0].to_string(),
                operator: NumericOp::Eq,
                value: val,
                upper: None,
                field_hint: hint,
            });
        }
    }

    constraints
}

/// Guess which field a numeric constraint refers to based on surrounding context.
///
/// Looks for price/budget/rating/weight keywords near the number.
fn guess_field_hint_from_context(query: &str, position: usize) -> Option<String> {
    // Look at a window around the match position
    let start = position.saturating_sub(30);
    let end = (position + 50).min(query.len());
    let context = &query[start..end];

    let price_keywords = [
        "price", "cost", "budget", "$", "dollar", "usd", "eur", "gbp",
    ];
    let rating_keywords = ["rating", "star", "score", "review"];
    let weight_keywords = ["weight", "kg", "lb", "gram", "oz"];
    let size_keywords = ["size", "length", "width", "height"];

    for kw in &price_keywords {
        if context.contains(kw) {
            return Some("price".to_string());
        }
    }
    for kw in &rating_keywords {
        if context.contains(kw) {
            return Some("rating".to_string());
        }
    }
    for kw in &weight_keywords {
        if context.contains(kw) {
            return Some("weight".to_string());
        }
    }
    for kw in &size_keywords {
        if context.contains(kw) {
            return Some("size".to_string());
        }
    }

    // If a $ sign is anywhere in the query, it's likely price-related
    if query.contains('$') {
        return Some("price".to_string());
    }

    None
}

/// Extract string enum constraints by matching query text against known schema values.
///
/// For each `string_filter` field with known enum values, checks if the query
/// mentions any of those values (or common synonyms).
fn extract_string_enum_constraints(
    query: &str,
    filter_properties: &HashMap<String, Vec<String>>,
) -> Vec<ExtractedConstraint> {
    let mut constraints = Vec::new();
    let lower = query.to_lowercase();

    for (field_name, enum_values) in filter_properties {
        for value in enum_values {
            let value_lower = value.to_lowercase();

            // Direct match: the enum value appears in the query
            if lower.contains(&value_lower) {
                constraints.push(ExtractedConstraint::StringEnum {
                    original_text: value.clone(),
                    matched_value: value.clone(),
                    field_name: field_name.clone(),
                });
                continue;
            }

            // Synonym matching for common patterns
            if let Some(matched) = match_synonym(&lower, &value_lower, field_name) {
                constraints.push(ExtractedConstraint::StringEnum {
                    original_text: matched,
                    matched_value: value.clone(),
                    field_name: field_name.clone(),
                });
            }
        }
    }

    constraints
}

/// Match common synonyms in queries to enum values.
///
/// Handles patterns like "men's" -> "male", "women's" -> "female", etc.
fn match_synonym(query: &str, enum_value: &str, _field_name: &str) -> Option<String> {
    let gender_mappings: Vec<(&[&str], &str)> = vec![
        (
            &["men's", "mens", "for men", "male", "man's", "boys", "boy's"],
            "male",
        ),
        (
            &[
                "women's",
                "womens",
                "for women",
                "female",
                "woman's",
                "girls",
                "girl's",
            ],
            "female",
        ),
        (
            &["unisex", "gender neutral", "for everyone", "all genders"],
            "unisex",
        ),
        (
            &[
                "kids",
                "children",
                "for kids",
                "child",
                "kid's",
                "children's",
            ],
            "kids",
        ),
    ];

    for (synonyms, target) in &gender_mappings {
        if *target == enum_value {
            for synonym in *synonyms {
                if query.contains(synonym) {
                    return Some(synonym.to_string());
                }
            }
        }
    }

    None
}

/// Extract boolean constraints from the query.
///
/// Matches patterns like "in stock" -> true, "on sale" -> true, etc.
fn extract_boolean_constraints(
    query: &str,
    schema_fields: &[FieldInfo],
) -> Vec<ExtractedConstraint> {
    let mut constraints = Vec::new();
    let lower = query.to_lowercase();

    let boolean_fields: Vec<&FieldInfo> = schema_fields
        .iter()
        .filter(|f| f.field_type == "boolean")
        .collect();

    if boolean_fields.is_empty() {
        return constraints;
    }

    // Common boolean patterns with their implied values
    let true_patterns = [
        "in stock",
        "available",
        "on sale",
        "active",
        "enabled",
        "verified",
        "featured",
        "published",
        "approved",
    ];
    let false_patterns = [
        "out of stock",
        "unavailable",
        "not on sale",
        "inactive",
        "disabled",
        "not verified",
        "unpublished",
    ];

    for pattern in &true_patterns {
        if lower.contains(pattern) {
            // Try to match to a boolean field name
            let hint = find_matching_boolean_field(pattern, &boolean_fields);
            constraints.push(ExtractedConstraint::Boolean {
                original_text: pattern.to_string(),
                value: true,
                field_hint: hint,
            });
        }
    }

    for pattern in &false_patterns {
        if lower.contains(pattern) {
            let hint = find_matching_boolean_field(pattern, &boolean_fields);
            constraints.push(ExtractedConstraint::Boolean {
                original_text: pattern.to_string(),
                value: false,
                field_hint: hint,
            });
        }
    }

    constraints
}

/// Try to match a boolean pattern to a specific boolean field name.
fn find_matching_boolean_field(pattern: &str, boolean_fields: &[&FieldInfo]) -> Option<String> {
    let stock_keywords = ["stock", "available", "availability"];
    let sale_keywords = ["sale", "discount", "offer"];
    let active_keywords = ["active", "enabled", "published"];

    for field in boolean_fields {
        let field_lower = field.name.to_lowercase();
        if stock_keywords.iter().any(|kw| pattern.contains(kw))
            && stock_keywords.iter().any(|kw| field_lower.contains(kw))
        {
            return Some(field.name.clone());
        }
        if sale_keywords.iter().any(|kw| pattern.contains(kw))
            && sale_keywords.iter().any(|kw| field_lower.contains(kw))
        {
            return Some(field.name.clone());
        }
        if active_keywords.iter().any(|kw| pattern.contains(kw))
            && active_keywords.iter().any(|kw| field_lower.contains(kw))
        {
            return Some(field.name.clone());
        }
    }

    // If only one boolean field exists, use it
    if boolean_fields.len() == 1 {
        return Some(boolean_fields[0].name.clone());
    }

    None
}

// ===== Field Matching =====

/// Match numeric constraints to specific number fields in the schema.
///
/// Heuristic: if one number field exists, all numeric constraints map to it.
/// If multiple exist, use field hints to disambiguate.
pub fn match_numeric_constraints(
    constraints: &[ExtractedConstraint],
    number_fields: &[String],
) -> Vec<(String, ExtractedConstraint)> {
    let mut matched = Vec::new();

    let numeric_constraints: Vec<&ExtractedConstraint> = constraints
        .iter()
        .filter(|c| matches!(c, ExtractedConstraint::Numeric { .. }))
        .collect();

    if numeric_constraints.is_empty() || number_fields.is_empty() {
        return matched;
    }

    // If only one number field, all numeric constraints map to it
    if number_fields.len() == 1 {
        for constraint in numeric_constraints {
            matched.push((number_fields[0].clone(), constraint.clone()));
        }
        return matched;
    }

    // Multiple number fields: use field hints to disambiguate
    for constraint in numeric_constraints {
        if let ExtractedConstraint::Numeric { field_hint, .. } = constraint {
            if let Some(hint) = field_hint {
                // Find the best matching field for this hint
                if let Some(field_name) = find_best_field_match(hint, number_fields) {
                    matched.push((field_name, constraint.clone()));
                    continue;
                }
            }
            // Fallback: try to match to the most common price-like field
            let price_like = number_fields.iter().find(|f| {
                let fl = f.to_lowercase();
                fl.contains("price") || fl.contains("cost") || fl.contains("amount")
            });
            if let Some(field) = price_like {
                matched.push((field.clone(), constraint.clone()));
            }
        }
    }

    matched
}

/// Find the best matching field name for a given hint.
fn find_best_field_match(hint: &str, fields: &[String]) -> Option<String> {
    let hint_lower = hint.to_lowercase();

    // Exact match
    if let Some(exact) = fields.iter().find(|f| f.to_lowercase() == hint_lower) {
        return Some(exact.clone());
    }

    // Contains match (e.g., hint "price" matches "xx_price" or "fullPrice")
    if let Some(contains) = fields
        .iter()
        .find(|f| f.to_lowercase().contains(&hint_lower))
    {
        return Some(contains.clone());
    }

    // Reverse contains (e.g., hint "product_price" matches field "price")
    if let Some(reverse) = fields
        .iter()
        .find(|f| hint_lower.contains(&f.to_lowercase()))
    {
        return Some(reverse.clone());
    }

    None
}

// ===== Formatting =====

/// Format extracted constraints as a human-readable string for the LLM prompt.
pub fn format_constraints_for_prompt(
    constraints: &[ExtractedConstraint],
    number_fields: &[String],
) -> String {
    if constraints.is_empty() {
        return "No constraints detected.".to_string();
    }

    let matched_numerics = match_numeric_constraints(constraints, number_fields);
    let mut lines = Vec::new();

    // Add matched numeric constraints
    for (field_name, constraint) in &matched_numerics {
        if let ExtractedConstraint::Numeric {
            operator,
            value,
            upper,
            original_text,
            ..
        } = constraint
        {
            let op_str = match operator {
                NumericOp::Lte => format!("lte {value}"),
                NumericOp::Gte => format!("gte {value}"),
                NumericOp::Eq => format!("eq {value}"),
                NumericOp::Between => {
                    if let Some(u) = upper {
                        format!("between {value} and {u}")
                    } else {
                        format!("gte {value}")
                    }
                }
            };
            lines.push(format!(
                "- {field_name}: {op_str} (from \"{original_text}\")"
            ));
        }
    }

    // Add string enum constraints
    for constraint in constraints {
        if let ExtractedConstraint::StringEnum {
            field_name,
            matched_value,
            original_text,
            ..
        } = constraint
        {
            lines.push(format!(
                "- {field_name}: \"{matched_value}\" (from \"{original_text}\", matched to enum value)"
            ));
        }
    }

    // Add boolean constraints
    for constraint in constraints {
        if let ExtractedConstraint::Boolean {
            value,
            field_hint,
            original_text,
            ..
        } = constraint
        {
            let field = field_hint.as_deref().unwrap_or("(unknown field)");
            lines.push(format!("- {field}: {value} (from \"{original_text}\")"));
        }
    }

    if lines.is_empty() {
        "No constraints detected.".to_string()
    } else {
        lines.join("\n")
    }
}

// ===== Validation & Injection (Tier 3) =====

/// Validate that the LLM-generated SearchParams contain all extracted constraints.
///
/// Returns any constraints that are missing from the search params.
pub fn validate_search_params(
    params: &SearchParams,
    constraints: &[ExtractedConstraint],
    number_fields: &[String],
) -> Vec<ExtractedConstraint> {
    let mut missing = Vec::new();
    let matched_numerics = match_numeric_constraints(constraints, number_fields);

    // Check numeric constraints
    for (field_name, constraint) in &matched_numerics {
        if !where_filter_has_field(&params.where_filter, field_name) {
            missing.push(constraint.clone());
        }
    }

    // Check string enum constraints
    for constraint in constraints {
        if let ExtractedConstraint::StringEnum { field_name, .. } = constraint {
            if !where_filter_has_field(&params.where_filter, field_name) {
                missing.push(constraint.clone());
            }
        }
    }

    // Check boolean constraints
    for constraint in constraints {
        if let ExtractedConstraint::Boolean {
            field_hint: Some(field),
            ..
        } = constraint
        {
            if !where_filter_has_field(&params.where_filter, field) {
                missing.push(constraint.clone());
            }
        }
    }

    missing
}

/// Check if a WhereFilter contains a filter for the given field name.
fn where_filter_has_field(filter: &WhereFilter, field_name: &str) -> bool {
    // Check direct field filters
    for (name, _) in &filter.filter_on_fields {
        if name == field_name {
            return true;
        }
    }

    // Check AND clauses
    if let Some(and_clauses) = &filter.and {
        for clause in and_clauses {
            if where_filter_has_field(clause, field_name) {
                return true;
            }
        }
    }

    // Check OR clauses
    if let Some(or_clauses) = &filter.or {
        for clause in or_clauses {
            if where_filter_has_field(clause, field_name) {
                return true;
            }
        }
    }

    false
}

/// Inject missing constraints directly into the SearchParams where filter.
///
/// This is the safety net: if the LLM failed to include a constraint,
/// we add it here before the search executes.
pub fn inject_constraints(
    params: &mut SearchParams,
    missing: &[ExtractedConstraint],
    number_fields: &[String],
) {
    let matched_numerics = match_numeric_constraints(missing, number_fields);

    // Inject numeric constraints
    for (field_name, constraint) in matched_numerics {
        if let ExtractedConstraint::Numeric {
            operator,
            value,
            upper,
            ..
        } = &constraint
        {
            let number_filter = match operator {
                NumericOp::Lte => NumberFilter::LessThanOrEqual(f64_to_number(*value)),
                NumericOp::Gte => NumberFilter::GreaterThanOrEqual(f64_to_number(*value)),
                NumericOp::Eq => NumberFilter::Equal(f64_to_number(*value)),
                NumericOp::Between => {
                    if let Some(u) = upper {
                        NumberFilter::Between((f64_to_number(*value), f64_to_number(*u)))
                    } else {
                        NumberFilter::GreaterThanOrEqual(f64_to_number(*value))
                    }
                }
            };

            params
                .where_filter
                .filter_on_fields
                .push((field_name, Filter::Number(number_filter)));
        }
    }

    // Inject string enum constraints
    for constraint in missing {
        if let ExtractedConstraint::StringEnum {
            field_name,
            matched_value,
            ..
        } = constraint
        {
            params
                .where_filter
                .filter_on_fields
                .push((field_name.clone(), Filter::String(matched_value.clone())));
        }
    }

    // Inject boolean constraints
    for constraint in missing {
        if let ExtractedConstraint::Boolean {
            value,
            field_hint: Some(field),
            ..
        } = constraint
        {
            params
                .where_filter
                .filter_on_fields
                .push((field.clone(), Filter::Bool(*value)));
        }
    }
}

/// Convert f64 to the Number type used by the search engine.
fn f64_to_number(value: f64) -> crate::types::Number {
    // Use i32 if the value is a whole number within range, otherwise f32
    if value.fract() == 0.0 && value >= i32::MIN as f64 && value <= i32::MAX as f64 {
        crate::types::Number::I32(value as i32)
    } else {
        crate::types::Number::F32(value as f32)
    }
}

// ===== Budget Detection =====

/// Check if the original query implies a shared budget across multiple sub-queries.
///
/// Returns true if the query has budget language AND multi-item coordination language.
pub fn has_shared_budget(query: &str) -> bool {
    let lower = query.to_lowercase();

    let budget_signals = [
        "budget",
        "total",
        "combined",
        "for both",
        "for all",
        "altogether",
        "in total",
        "max for",
        "spend",
    ];

    // Allow optional filler words (e.g. "of", "around", "about") between the
    // budget keyword and the currency/amount so "Budget of $500" matches.
    let has_budget_number = Regex::new(
        r"(?i)(?:under|below|max|budget|up\s+to|no\s+more\s+than)\s+(?:\w+\s+)*?(?:\$|usd|eur|gbp|£|€)?\s*\d+",
    )
    .expect("Valid regex")
    .is_match(&lower);

    let has_coordination = budget_signals.iter().any(|s| lower.contains(s));

    has_budget_number && has_coordination
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal SearchParams for testing via JSON deserialization.
    fn empty_search_params() -> SearchParams {
        serde_json::from_str::<SearchParams>(r#"{"term": "", "mode": "fulltext"}"#)
            .expect("Valid test SearchParams")
    }

    #[test]
    fn test_extract_lte_constraint() {
        let constraints = extract_numeric_constraints("shoes under $100");
        assert_eq!(constraints.len(), 1);
        if let ExtractedConstraint::Numeric {
            operator, value, ..
        } = &constraints[0]
        {
            assert_eq!(*operator, NumericOp::Lte);
            assert!((value - 100.0).abs() < 0.01);
        } else {
            panic!("Expected numeric constraint");
        }
    }

    #[test]
    fn test_extract_gte_constraint() {
        let constraints = extract_numeric_constraints("products over $50");
        assert_eq!(constraints.len(), 1);
        if let ExtractedConstraint::Numeric {
            operator, value, ..
        } = &constraints[0]
        {
            assert_eq!(*operator, NumericOp::Gte);
            assert!((value - 50.0).abs() < 0.01);
        } else {
            panic!("Expected numeric constraint");
        }
    }

    #[test]
    fn test_extract_between_constraint() {
        let constraints = extract_numeric_constraints("shoes between $50 and $150");
        assert_eq!(constraints.len(), 1);
        if let ExtractedConstraint::Numeric {
            operator,
            value,
            upper,
            ..
        } = &constraints[0]
        {
            assert_eq!(*operator, NumericOp::Between);
            assert!((value - 50.0).abs() < 0.01);
            assert!((upper.unwrap() - 150.0).abs() < 0.01);
        } else {
            panic!("Expected numeric constraint");
        }
    }

    #[test]
    fn test_extract_eq_constraint() {
        let constraints = extract_numeric_constraints("exactly $75");
        assert_eq!(constraints.len(), 1);
        if let ExtractedConstraint::Numeric {
            operator, value, ..
        } = &constraints[0]
        {
            assert_eq!(*operator, NumericOp::Eq);
            assert!((value - 75.0).abs() < 0.01);
        } else {
            panic!("Expected numeric constraint");
        }
    }

    #[test]
    fn test_extract_string_enum_direct_match() {
        let mut filter_props = HashMap::new();
        filter_props.insert(
            "category".to_string(),
            vec![
                "shoes".to_string(),
                "boots".to_string(),
                "sandals".to_string(),
            ],
        );

        let constraints = extract_string_enum_constraints("I want running shoes", &filter_props);
        assert_eq!(constraints.len(), 1);
        if let ExtractedConstraint::StringEnum {
            matched_value,
            field_name,
            ..
        } = &constraints[0]
        {
            assert_eq!(matched_value, "shoes");
            assert_eq!(field_name, "category");
        } else {
            panic!("Expected string enum constraint");
        }
    }

    #[test]
    fn test_extract_string_enum_synonym_match() {
        let mut filter_props = HashMap::new();
        filter_props.insert(
            "gender".to_string(),
            vec![
                "male".to_string(),
                "female".to_string(),
                "unisex".to_string(),
            ],
        );

        let constraints = extract_string_enum_constraints("men's basketball shoes", &filter_props);
        assert_eq!(constraints.len(), 1);
        if let ExtractedConstraint::StringEnum {
            matched_value,
            field_name,
            ..
        } = &constraints[0]
        {
            assert_eq!(matched_value, "male");
            assert_eq!(field_name, "gender");
        } else {
            panic!("Expected string enum constraint");
        }
    }

    #[test]
    fn test_extract_boolean_constraint() {
        let schema = vec![FieldInfo {
            name: "in_stock".to_string(),
            field_type: "boolean".to_string(),
        }];
        let constraints = extract_boolean_constraints("show me items in stock", &schema);
        assert_eq!(constraints.len(), 1);
        if let ExtractedConstraint::Boolean { value, .. } = &constraints[0] {
            assert!(*value);
        } else {
            panic!("Expected boolean constraint");
        }
    }

    #[test]
    fn test_match_numeric_single_field() {
        let constraints = vec![ExtractedConstraint::Numeric {
            original_text: "under $100".to_string(),
            operator: NumericOp::Lte,
            value: 100.0,
            upper: None,
            field_hint: Some("price".to_string()),
        }];
        let fields = vec!["price".to_string()];
        let matched = match_numeric_constraints(&constraints, &fields);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].0, "price");
    }

    #[test]
    fn test_match_numeric_multiple_fields_with_hint() {
        let constraints = vec![
            ExtractedConstraint::Numeric {
                original_text: "under $100".to_string(),
                operator: NumericOp::Lte,
                value: 100.0,
                upper: None,
                field_hint: Some("price".to_string()),
            },
            ExtractedConstraint::Numeric {
                original_text: "at least 4".to_string(),
                operator: NumericOp::Gte,
                value: 4.0,
                upper: None,
                field_hint: Some("rating".to_string()),
            },
        ];
        let fields = vec!["price".to_string(), "rating".to_string()];
        let matched = match_numeric_constraints(&constraints, &fields);
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn test_format_constraints_for_prompt() {
        let constraints = vec![
            ExtractedConstraint::Numeric {
                original_text: "under $100".to_string(),
                operator: NumericOp::Lte,
                value: 100.0,
                upper: None,
                field_hint: Some("price".to_string()),
            },
            ExtractedConstraint::StringEnum {
                original_text: "men's".to_string(),
                matched_value: "male".to_string(),
                field_name: "gender".to_string(),
            },
        ];
        let fields = vec!["price".to_string()];
        let result = format_constraints_for_prompt(&constraints, &fields);
        assert!(result.contains("price: lte 100"));
        assert!(result.contains("gender: \"male\""));
    }

    #[test]
    fn test_has_shared_budget() {
        assert!(has_shared_budget(
            "I need shoes and pants, max $200 for both"
        ));
        assert!(has_shared_budget(
            "Budget of $500 total for a laptop and monitor"
        ));
        assert!(!has_shared_budget("shoes under $100"));
        assert!(!has_shared_budget("find me a nice laptop"));
    }

    #[test]
    fn test_validate_missing_constraints() {
        let constraints = vec![ExtractedConstraint::Numeric {
            original_text: "under $100".to_string(),
            operator: NumericOp::Lte,
            value: 100.0,
            upper: None,
            field_hint: Some("price".to_string()),
        }];
        let fields = vec!["price".to_string()];

        // Empty where filter means the constraint is missing
        let params = empty_search_params();
        let missing = validate_search_params(&params, &constraints, &fields);
        assert_eq!(missing.len(), 1);
    }

    #[test]
    fn test_inject_numeric_constraint() {
        let missing = vec![ExtractedConstraint::Numeric {
            original_text: "under $100".to_string(),
            operator: NumericOp::Lte,
            value: 100.0,
            upper: None,
            field_hint: Some("price".to_string()),
        }];
        let fields = vec!["price".to_string()];

        let mut params = empty_search_params();
        inject_constraints(&mut params, &missing, &fields);

        assert_eq!(params.where_filter.filter_on_fields.len(), 1);
        assert_eq!(params.where_filter.filter_on_fields[0].0, "price");
    }

    #[test]
    fn test_inject_string_enum_constraint() {
        let missing = vec![ExtractedConstraint::StringEnum {
            original_text: "men's".to_string(),
            matched_value: "male".to_string(),
            field_name: "gender".to_string(),
        }];
        let fields: Vec<String> = vec![];

        let mut params = empty_search_params();
        inject_constraints(&mut params, &missing, &fields);

        assert_eq!(params.where_filter.filter_on_fields.len(), 1);
        assert_eq!(params.where_filter.filter_on_fields[0].0, "gender");
    }

    #[test]
    fn test_price_hint_from_dollar_sign() {
        let constraints = extract_numeric_constraints("shoes under $100");
        assert_eq!(constraints.len(), 1);
        if let ExtractedConstraint::Numeric { field_hint, .. } = &constraints[0] {
            assert_eq!(field_hint.as_deref(), Some("price"));
        }
    }

    #[test]
    fn test_at_least_with_star_rating() {
        let constraints = extract_numeric_constraints("at least 4 star rating");
        assert_eq!(constraints.len(), 1);
        if let ExtractedConstraint::Numeric {
            operator,
            value,
            field_hint,
            ..
        } = &constraints[0]
        {
            assert_eq!(*operator, NumericOp::Gte);
            assert!((value - 4.0).abs() < 0.01);
            assert_eq!(field_hint.as_deref(), Some("rating"));
        }
    }

    #[test]
    fn test_no_constraints_in_simple_query() {
        let constraints = extract_numeric_constraints("what are good basketball shoes");
        assert!(constraints.is_empty());
    }

    #[test]
    fn test_full_extract_constraints() {
        let mut filter_props = HashMap::new();
        filter_props.insert(
            "gender".to_string(),
            vec!["male".to_string(), "female".to_string()],
        );

        let schema = vec![
            FieldInfo {
                name: "price".to_string(),
                field_type: "number".to_string(),
            },
            FieldInfo {
                name: "gender".to_string(),
                field_type: "string_filter".to_string(),
            },
        ];

        let constraints =
            extract_constraints("men's basketball shoes under $100", &schema, &filter_props);

        // Should have at least: numeric (under $100) + string enum (men's -> male)
        assert!(constraints.len() >= 2);

        let has_numeric = constraints
            .iter()
            .any(|c| matches!(c, ExtractedConstraint::Numeric { .. }));
        let has_string = constraints
            .iter()
            .any(|c| matches!(c, ExtractedConstraint::StringEnum { .. }));
        assert!(has_numeric);
        assert!(has_string);
    }
}
