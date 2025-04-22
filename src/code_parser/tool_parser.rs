use oxc::ast::ast::{ExportDefaultDeclaration, Expression, ObjectPropertyKind, PropertyKey};
use oxc::ast::Visit;
use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_span::SourceType;

/// ToolValidator is used to validate the code of a given tool.
/// Every tool must:
/// 1. Use `export default`.
/// 2. Export an object.
/// 3. The object must have a single function.
/// 4. The function must have a valid name.
pub struct ToolValidator {
    pub has_export_default: bool,
    pub exports_object: bool,
    pub exports_single_function: bool,
    pub exported_function_name: Option<String>,
    pub reason: Option<String>,
}

impl ToolValidator {
    pub fn new() -> Self {
        ToolValidator {
            has_export_default: false,
            exports_object: false,
            exports_single_function: false,
            exported_function_name: None,
            reason: None,
        }
    }
}

impl<'a> Visit<'a> for ToolValidator {
    fn visit_export_default_declaration(&mut self, export: &ExportDefaultDeclaration<'a>) {
        self.has_export_default = true;

        if let Some(expr) = export.declaration.as_expression() {
            if let Expression::ObjectExpression(obj_expr) = expr {
                self.exports_object = true;

                if obj_expr.properties.len() == 1 {
                    if let Some(ObjectPropertyKind::ObjectProperty(property)) =
                        obj_expr.properties.first()
                    {
                        match &property.key {
                            PropertyKey::StaticIdentifier(ident) => match &property.value {
                                Expression::FunctionExpression(_)
                                | Expression::ArrowFunctionExpression(_) => {
                                    self.exports_single_function = true;
                                    self.exported_function_name = Some(ident.name.to_string());
                                }
                                _ => {
                                    self.reason = Some(
                                        "Exported property's value is not a function.".to_string(),
                                    );
                                }
                            },
                            _ => {
                                self.reason = Some(
                                    "Exported property's key is not an identifier.".to_string(),
                                );
                            }
                        }
                    } else {
                        self.reason = Some("Exported object must contain a property.".to_string());
                    }
                } else {
                    self.reason =
                        Some("Exported object must have exactly one property.".to_string());
                }
            } else {
                self.reason = Some("Export default must be an object expression.".to_string());
            }
        } else {
            self.reason = Some("Export default is not an expression.".to_string());
        }
    }
}

pub fn validate_js_exports(
    source_code: &str,
) -> Result<(bool, Option<String>, Option<String>), String> {
    let source_type = SourceType::default().with_module(true);

    let allocator = Allocator::default();
    let parser = Parser::new(&allocator, source_code, source_type);

    let parse_result = parser.parse();

    if parse_result.errors.is_empty() {
        let program = parse_result.program;
        let mut validator = ToolValidator::new();
        validator.visit_program(&program);

        let is_valid = validator.has_export_default
            && validator.exports_object
            && validator.exports_single_function;

        if is_valid {
            Ok((true, validator.exported_function_name, None))
        } else {
            let reason = validator.reason.or_else(|| {
                if !validator.has_export_default {
                    Some("Missing `export default`.".to_string())
                } else if !validator.exports_object {
                    Some("Exported default is not an object.".to_string())
                } else if !validator.exports_single_function {
                    Some("Exported object must contain a single function.".to_string())
                } else {
                    Some("Unknown reason.".to_string())
                }
            });

            Ok((false, None, reason))
        }
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
        let (is_valid, name, reason) = result.unwrap();
        assert!(is_valid);
        assert_eq!(name, Some("myFunction".to_string()));
        assert!(reason.is_none());
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
        let (is_valid, name, reason) = result.unwrap();
        assert!(!is_valid);
        assert_eq!(name, None);
        assert_eq!(reason, Some("Missing `export default`.".to_string()));
    }

    #[test]
    fn export_not_an_object() {
        let js = r#"
            export default 123;
        "#;

        let result = validate_js_exports(js);
        assert!(result.is_ok());
        let (is_valid, _, reason) = result.unwrap();
        assert!(!is_valid);
        assert_eq!(
            reason,
            Some("Export default must be an object expression.".to_string())
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
        let (is_valid, _, reason) = result.unwrap();
        assert!(!is_valid);
        assert_eq!(
            reason,
            Some("Exported object must have exactly one property.".to_string())
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
        let (is_valid, _, reason) = result.unwrap();
        assert!(!is_valid);
        assert_eq!(
            reason,
            Some("Exported property's value is not a function.".to_string())
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
        let (is_valid, _, reason) = result.unwrap();
        assert!(!is_valid);
        assert_eq!(
            reason,
            Some("Exported property's key is not an identifier.".to_string())
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
