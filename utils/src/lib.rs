use anyhow::Result;
use regex::Regex;
use serde_json::Value;

pub fn parse_json_safely(input_str: &str) -> Result<Value> {
    if let Ok(parsed) = serde_json::from_str(input_str.trim()) {
        return Ok(parsed);
    }

    let trimmed_input = input_str.trim();

    let json_regex = Regex::new(r"```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*\}|\[[\s\S]*\])")
        .expect("Failed to compile regex");

    if let Some(captures) = json_regex.captures(trimmed_input) {
        if let Some(json_string) = captures.get(1).or_else(|| captures.get(2)) {
            if let Ok(parsed) = serde_json::from_str(json_string.as_str().trim()) {
                return Ok(parsed);
            }
        }
    }

    anyhow::bail!("Failed to parse JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_valid_json() {
        let input = r#"{"key": "value"}"#;
        let result = parse_json_safely(input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"key": "value"}));
    }

    #[test]
    fn test_parse_json_in_code_block() {
        let input = r#"```json
        {
            "name": "John",
            "age": 30
        }
        ```"#;
        let result = parse_json_safely(input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"name": "John", "age": 30}));
    }

    #[test]
    fn test_parse_json_without_language_specifier() {
        let input = r#"```
        {
            "status": "ok"
        }
        ```"#;
        let result = parse_json_safely(input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"status": "ok"}));
    }

    #[test]
    fn test_parse_partial_json() {
        let input = r#"
        Some text before
        {
            "foo": "bar"
        }
        Some text after
        "#;
        let result = parse_json_safely(input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"foo": "bar"}));
    }

    #[test]
    fn test_invalid_json_returns_error() {
        let input = r#"Invalid JSON string"#;
        let result = parse_json_safely(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_incomplete_json_structure() {
        let input = r#"
        {
            "incomplete": true,
        "#;
        let result = parse_json_safely(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_json_in_plain_text_with_extra_text() {
        let input = r#"
        Here is some text and then:
        {
            "key": "value"
        }
        Followed by more text.
        "#;
        let result = parse_json_safely(input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"key": "value"}));
    }
}
