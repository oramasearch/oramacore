use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use anyhow::Error;
use deno_core::{FastString, JsRuntime, RuntimeOptions};
use deno_core::v8::{Global, Value};
use serde::Serialize;
use strum_macros::Display;
use twox_hash::{XxHash64};

#[derive(Display)]
pub enum Context {
    EmbeddingsPropertiesSelector,
    SourcesNormalizator,
    RankingMapReduce,
    Anonymous
}

pub struct JavaScript {
    pub runtime: JsRuntime,
    cached_functions: HashMap<String, String, BuildHasherDefault<XxHash64>>,
}

impl JavaScript {
    pub fn new() -> Self {
        Self {
            runtime: JsRuntime::new(RuntimeOptions::default()),
            cached_functions: HashMap::default()
        }
    }

    pub async fn eval<T: Serialize>(&mut self, context: Context, code: String, input: T) -> Result<Global<Value>, Error> {
        let cache_key = format!("{}:{}", context, &code);
        self.ensure_function_cached(&context, &code).await?;

        let func_name = self.cached_functions.get(&cache_key).unwrap();
        let input_json = serde_json::to_string(&input)?;

        let execute_code = format!(
            r#"
                (async () => {{
                    const input = JSON.parse('{}');
                    const fn = await globalThis.{};
                    return await fn(input);
                }})();
            "#,
            input_json.replace('\'', "\\'"),
            func_name
        );

        let promise = self.runtime.execute_script("[exec]", FastString::from(execute_code))?;
        let result = self.runtime.resolve(promise).await?;
        Ok(result)
    }

    async fn ensure_function_cached(&mut self, context: &Context, code: &str) -> Result<(), Error> {
        let cache_key = format!("{}:{}", context, code);

        if !self.cached_functions.contains_key(&cache_key) {
            self.runtime.execute_script("[decl]", FastString::from(code.to_string()))?;

            let func_name = format!("func_{}", self.cached_functions.len());
            let cache_code = format!(
                r#"
                    globalThis.{} = (async () => {{
                        const module = await import('[decl]');
                        return module.default;
                    }})();
                "#,
                func_name
            );

            let promise = self.runtime.execute_script("[cache]", FastString::from(cache_code))?;
            self.runtime.resolve(promise).await?;
            self.cached_functions.insert(cache_key, func_name);
        }

        Ok(())
    }

    pub async fn clear_cache(&mut self) {
        for func_name in self.cached_functions.values() {
            let cleanup_code = format!("delete globalThis.{};", func_name);
            let _ = self.runtime.execute_script(
                "[cleanup]",
                FastString::from(cleanup_code)
            );
        }
        self.cached_functions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use deno_core::v8;

    fn to_i32(global: Global<Value>) -> i32 {
        let isolate = &mut v8::Isolate::new(Default::default());
        let handle_scope = &mut v8::HandleScope::new(isolate);
        let context = v8::Context::new(handle_scope, v8::ContextOptions::default());
        let scope = &mut v8::ContextScope::new(handle_scope, context);
        let local = v8::Local::new(scope, global);
        local.to_integer(scope).unwrap().value() as i32
    }

    fn to_string(global: Global<Value>) -> String {
        let isolate = &mut v8::Isolate::new(Default::default());
        let handle_scope = &mut v8::HandleScope::new(isolate);
        let context = v8::Context::new(handle_scope, v8::ContextOptions::default());
        let scope = &mut v8::ContextScope::new(handle_scope, context);
        let local = v8::Local::new(scope, global);
        local.to_string(scope).unwrap().to_rust_string_lossy(scope)
    }

    #[tokio::test]
    async fn test_basic_function_execution() -> Result<(), Error> {
        let mut js = JavaScript::new();
        let code = r#"
            export default function(input) {
                return input.x * 2;
            }
        "#.to_string();

        let result = js.eval(Context::Anonymous, code, json!({"x": 21})).await?;
        assert_eq!(to_i32(result), 42);
        Ok(())
    }

    #[tokio::test]
    async fn test_function_caching() -> Result<(), Error> {
        let mut js = JavaScript::new();
        let code = r#"
            export default function(input) {
                return input.value + 1;
            }
        "#.to_string();

        let result1 = js.eval(Context::Anonymous, code.clone(), json!({"value": 41})).await?;
        assert_eq!(to_i32(result1), 42);
        assert_eq!(js.cached_functions.len(), 1);

        let result2 = js.eval(Context::Anonymous, code.clone(), json!({"value": 99})).await?;
        assert_eq!(to_i32(result2), 100);
        assert_eq!(js.cached_functions.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_different_contexts() -> Result<(), Error> {
        let mut js = JavaScript::new();
        let code = r#"
            export default function(input) {
                return input.x * 2;
            }
        "#.to_string();

        let result1 = js.eval(Context::EmbeddingsPropertiesSelector, code.clone(), json!({"x": 21})).await?;
        let result2 = js.eval(Context::SourcesNormalizator, code.clone(), json!({"x": 21})).await?;

        assert_eq!(to_i32(result1), 42);
        assert_eq!(to_i32(result2), 42);

        assert_eq!(js.cached_functions.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_code_modification() -> Result<(), Error> {
        let mut js = JavaScript::new();
        let code1 = r#"
            export default function(input) {
                return input.x * 2;
            }
        "#.to_string();

        let code2 = r#"
            export default function(input) {
                return input.x * 3;
            }
        "#.to_string();

        let result1 = js.eval(Context::Anonymous, code1, json!({"x": 21})).await?;
        assert_eq!(to_i32(result1), 42);

        let result2 = js.eval(Context::Anonymous, code2, json!({"x": 21})).await?;
        assert_eq!(to_i32(result2), 63);

        assert_eq!(js.cached_functions.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_complex_return_types() -> Result<(), Error> {
        let mut js = JavaScript::new();
        let code = r#"
            export default function(input) {
                return {
                    message: `Hello ${input.name}!`,
                    count: input.count + 1
                };
            }
        "#.to_string();

        let result = js.eval(
            Context::Anonymous,
            code,
            json!({
                "name": "World",
                "count": 41
            })
        ).await?;

        let result_str = to_string(result);
        assert!(result_str.contains("Hello") && result_str.contains("World"));
        Ok(())
    }

    #[tokio::test]
    async fn test_error_handling() {
        let mut js = JavaScript::new();
        let invalid_code = r#"
            export default function(input) {
                throw new Error('Test error');
            }
        "#.to_string();

        let result = js.eval(Context::Anonymous, invalid_code, json!({})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_async_function() -> Result<(), Error> {
        let mut js = JavaScript::new();
        let code = r#"
        export default async function(input) {
            // Simulate async work
            await new Promise(resolve => setTimeout(resolve, 100));
            return input.x * 2;
        }
    "#.to_string();

        let result = js.eval(Context::Anonymous, code, json!({"x": 21})).await?;
        assert_eq!(to_i32(result), 42);
        Ok(())
    }

    #[tokio::test]
    async fn test_promise_chain() -> Result<(), Error> {
        let mut js = JavaScript::new();
        let code = r#"
        export default async function(input) {
            const value = await Promise.resolve(input.x)
                .then(x => x * 2)
                .then(x => x + 1);
            return value;
        }
    "#.to_string();

        let result = js.eval(Context::Anonymous, code, json!({"x": 20})).await?;
        assert_eq!(to_i32(result), 41);
        Ok(())
    }
}