use crate::questions_generation::prompts::{
    get_questions_generation_prompt, QUESTIONS_GENERATION_SYSTEM_PROMPT,
};
use crate::LocalLLM;
use anyhow::{Context, Result};
use mistralrs::{IsqType, TextMessageRole, TextMessages, TextModelBuilder};
use serde_json::Value;
use textwrap::dedent;
use utils::parse_json_safely;

pub async fn generate_questions(context: String) -> Result<Vec<String>> {
    let model = LocalLLM::Phi3_5MiniInstruct.try_new().await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            dedent(QUESTIONS_GENERATION_SYSTEM_PROMPT),
        )
        .add_message(
            TextMessageRole::User,
            get_questions_generation_prompt(context),
        );

    let response = model
        .send_chat_request(messages)
        .await
        .context("Failed to send chat request")?;

    if let Some(content) = response
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone())
    {
        match parse_json_safely(content) {
            Ok(Value::Array(json_array)) => {
                let questions: Vec<String> = json_array
                    .iter()
                    .filter_map(|val| val.as_str().map(|s| s.to_string()))
                    .collect();
                Ok(questions)
            }
            Ok(_) => anyhow::bail!("Parsed content is not an array of strings"),
            Err(e) => Err(e).context("Failed to parse response content as JSON"),
        }
    } else {
        anyhow::bail!("No content in the response");
    }
}
