use anyhow::Result;
use quick_js::{Context, ExecutionError};
pub struct PropertiesSelector {
    js_engine: Context,
}

impl PropertiesSelector {
    pub fn try_new() -> Result<Self> {
        Ok(PropertiesSelector {
            js_engine: Context::new()?,
        })
    }

    pub fn eval(&self, code: String, doc: String) -> Result<Vec<String>, ExecutionError> {
        let selector_code = format!("selectProperties({})", doc);
        let full_code = format!("{}\n{}", code, selector_code);

        self.js_engine.eval_as::<Vec<String>>(&full_code)
    }
}
