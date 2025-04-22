use oxc::ast::ast::{ExportDefaultDeclaration, Expression, ObjectPropertyKind, PropertyKey};
use oxc::ast::Visit;
use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_span::SourceType;

#[derive(Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub function_name: Option<String>,
    pub error_reason: Option<String>,
}

/// Validates JavaScript code to ensure it follows the expected tool format.
///
/// A valid tool must:
/// 1. Use `export default` to export a value
/// 2. The exported value must be an object literal
/// 3. The object must contain exactly one property
/// 4. That property's value must be a function (regular or arrow)
pub struct ToolValidator {
    has_export_default: bool,
    exports_object: bool,
    exports_single_function: bool,
    exported_function_name: Option<String>,
    error_reason: Option<String>,
}

impl ToolValidator {
    pub fn new() -> Self {
        Self {
            has_export_default: false,
            exports_object: false,
            exports_single_function: false,
            exported_function_name: None,
            error_reason: None,
        }
    }

    fn set_error_if_none(&mut self, reason: &str) {
        if self.error_reason.is_none() {
            self.error_reason = Some(reason.to_string());
        }
    }

    pub fn result(self) -> ValidationResult {
        let is_valid =
            self.has_export_default && self.exports_object && self.exports_single_function;

        let error_reason = if !is_valid && self.error_reason.is_none() {
            if !self.has_export_default {
                Some("Missing `export default`".to_string())
            } else if !self.exports_object {
                Some("Exported default is not an object".to_string())
            } else if !self.exports_single_function {
                Some("Exported object must contain exactly one function".to_string())
            } else {
                Some("Unknown validation error".to_string())
            }
        } else {
            self.error_reason
        };

        ValidationResult {
            is_valid,
            function_name: self.exported_function_name,
            error_reason,
        }
    }

    /// Validates if the export is a valid function property
    fn validate_property(&mut self, property: &ObjectPropertyKind) -> bool {
        if let ObjectPropertyKind::ObjectProperty(prop) = property {
            // Check that key is a static identifier
            if let PropertyKey::StaticIdentifier(ident) = &prop.key {
                // Check that value is a function
                match &prop.value {
                    Expression::FunctionExpression(_) | Expression::ArrowFunctionExpression(_) => {
                        self.exports_single_function = true;
                        self.exported_function_name = Some(ident.name.to_string());
                        return true;
                    }
                    _ => self.set_error_if_none("Exported property's value is not a function"),
                }
            } else {
                self.set_error_if_none("Exported property's key is not an identifier");
            }
        } else {
            self.set_error_if_none("Exported object property has incorrect format");
        }
        false
    }
}

impl<'a> Visit<'a> for ToolValidator {
    fn visit_export_default_declaration(&mut self, export: &ExportDefaultDeclaration<'a>) {
        self.has_export_default = true;

        // Extract the expression from the export declaration
        match export.declaration.as_expression() {
            Some(Expression::ObjectExpression(obj_expr)) => {
                self.exports_object = true;

                // Check object has exactly one property
                if obj_expr.properties.len() != 1 {
                    self.set_error_if_none("Exported object must have exactly one property");
                    return;
                }

                // Validate the single property
                if let Some(property) = obj_expr.properties.first() {
                    self.validate_property(property);
                } else {
                    self.set_error_if_none("Exported object has no properties");
                }
            }
            Some(_) => self.set_error_if_none("Export default must be an object expression"),
            None => self.set_error_if_none("Export default is not an expression"),
        }
    }
}

pub fn validate_js_exports(source_code: &str) -> Result<ValidationResult, String> {
    // Configure the parser for ECMAScript modules
    let source_type = SourceType::default().with_module(true);
    let allocator = Allocator::default();
    let parser = Parser::new(&allocator, source_code, source_type);
    let parse_result = parser.parse();

    if parse_result.errors.is_empty() {
        let mut validator = ToolValidator::new();
        validator.visit_program(&parse_result.program);

        Ok(validator.result())
    } else {
        Err("Failed to parse JavaScript source code".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_tool_export() {
        let js = r#"
            export default {
                myFunction: function() {
                    console.log("Hello");
                }
            };
        "#;

        let result = validate_js_exports(js);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.is_valid);
        assert_eq!(validation.function_name, Some("myFunction".to_string()));
        assert_eq!(validation.error_reason, None);
    }

    #[test]
    fn missing_export_default() {
        let js = r#"
            const tool = {
                myFunction: function() {}
            };
        "#;

        let result = validate_js_exports(js);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(!validation.is_valid);
        assert_eq!(validation.function_name, None);
        assert_eq!(
            validation.error_reason,
            Some("Missing `export default`".to_string())
        );
    }

    #[test]
    fn export_not_an_object() {
        let js = r#"
            export default 123;
        "#;

        let result = validate_js_exports(js);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(!validation.is_valid);
        assert_eq!(
            validation.error_reason,
            Some("Export default must be an object expression".to_string())
        );
    }

    #[test]
    fn object_has_multiple_properties() {
        let js = r#"
            export default {
                a: function() {},
                b: function() {}
            };
        "#;

        let result = validate_js_exports(js);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(!validation.is_valid);
        assert_eq!(
            validation.error_reason,
            Some("Exported object must have exactly one property".to_string())
        );
    }

    #[test]
    fn property_is_not_function() {
        let js = r#"
            export default {
                notAFunction: 42
            };
        "#;

        let result = validate_js_exports(js);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(!validation.is_valid);
        assert_eq!(
            validation.error_reason,
            Some("Exported property's value is not a function".to_string())
        );
    }

    #[test]
    fn property_key_is_not_identifier() {
        let js = r#"
            export default {
                ["computed"]: function() {}
            };
        "#;

        let result = validate_js_exports(js);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(!validation.is_valid);
        assert_eq!(
            validation.error_reason,
            Some("Exported property's key is not an identifier".to_string())
        );
    }

    #[test]
    fn invalid_syntax() {
        let js = r#"
            export default {
                func() {
        "#;

        let result = validate_js_exports(js);
        assert!(result.is_err());
        assert_eq!(
            result.err(),
            Some("Failed to parse JavaScript source code".to_string())
        );
    }
}
