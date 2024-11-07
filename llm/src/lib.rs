use anyhow::{Context, Result};
use async_once_cell::OnceCell;
use async_std::sync::RwLock;
use mistralrs::{IsqType, Model, TextModelBuilder};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use strum::{Display, EnumIter};

pub mod content_expander;
pub mod conversational;
pub mod questions_generation;

static MODELS: OnceCell<RwLock<HashMap<LocalLLM, Arc<Model>>>> = OnceCell::new();

#[derive(Serialize, EnumIter, Eq, PartialEq, Hash, Clone, Display)]
pub enum LocalLLM {
    #[serde(rename = "microsoft/Phi-3.5-mini-instruct")]
    #[strum(serialize = "microsoft/Phi-3.5-mini-instruct")]
    Phi3_5MiniInstruct,

    #[serde(rename = "microsoft/Phi-3.5-vision-instruct")]
    #[strum(serialize = "microsoft/Phi-3.5-vision-instruct")]
    Phi3_5VisionInstruct,
}

impl LocalLLM {
    async fn try_new(&self) -> Result<Arc<Model>> {
        MODELS
            .get_or_init(async {
                let mut models_map = HashMap::new();
                let model = TextModelBuilder::new(self)
                    .with_isq(IsqType::Q8_0)
                    .with_logging()
                    .build()
                    .await
                    .with_context(|| "Failed to build the text model")
                    .unwrap();

                models_map.insert(self.clone(), Arc::new(model));
                RwLock::new(models_map)
            })
            .await;

        let models = MODELS.get().unwrap().read().await;
        models
            .get(self)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Model not found"))
    }
}
