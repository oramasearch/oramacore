use std::{env::var, fs};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::fs_utils::get_files;
use async_openai::{
    config::OpenAIConfig,
    types::{CreateCompletionRequest, CreateCompletionRequestArgs},
    Client,
};

struct CodeDescriptionGenerator {
    cache_folder: String,
    client: Client<OpenAIConfig>,
}
impl CodeDescriptionGenerator {
    fn new(cache_folder: String, client: Client<OpenAIConfig>) -> Self {
        let _ = fs::create_dir_all(&cache_folder);

        Self {
            cache_folder,
            client,
        }
    }

    async fn describe_as_text(&self, request: CreateCompletionRequest) -> String {
        let serialized = serde_json::to_string(&request).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        let result = hasher.finalize();
        let hash = format!("{:x}", result);

        if let Ok(content) = fs::read_to_string(format!("{}/{}", self.cache_folder, hash)) {
            return content;
        }

        let response = self.client.completions().create(request).await.unwrap();

        let text: String = response.choices.into_iter().map(|c| c.text).collect();

        fs::write(format!("{}/{}", self.cache_folder, hash), text.clone()).unwrap();

        text
    }
}

pub async fn parse_example(path: &str) -> Vec<Value> {
    let all_files = get_files(path.parse().unwrap(), vec!["tsx".to_string()]);

    let client =
        Client::with_config(OpenAIConfig::new().with_api_key(var("OPEN_API_KEY").unwrap()));
    let example_cache_dir = "./example_cache";
    let generator = &CodeDescriptionGenerator::new(example_cache_dir.to_string(), client);

    let futures: Vec<_> = all_files
        .into_iter()
        .map(|file| async move {
            let file_path = file.path();
            let content = fs::read_to_string(file_path.clone()).unwrap();

            let request = CreateCompletionRequestArgs::default()
                .model("gpt-3.5-turbo-instruct")
                .prompt(format!(r#"You are a bot that has to describe what the following code produce. Suppose the receiver know everything about Typescript and React.
- You can use jargon language
- Explain only what the user can do with the output of the code
- Starts with "In this example"
- Use only one paragraph
- Be concise
- Don't split the explaination in multiple paragraph.
- Don't explain how the code is structured.
- Don't describe the import.
- Don't describe what TanStack or React are.

```tsx
{content}
```
"#))
                .max_tokens(1024_u32)
                .temperature(0.0)
                .build()
                .unwrap();

            let description = generator.describe_as_text(request).await;

            let id = file_path.to_string_lossy().to_string();
            let example_name = id.replace(path, "");
            let example_name = example_name.split("/")
                .skip(1)
                .take(1)
                .next()
                .unwrap();
            let title = example_name.replace("-", " ");
            let title = capitalize(&title);
            let sub_path = format!("examples/react/{example_name}");

            serde_json::json!({
                "type": "example",
                "id": id,
                "code": content,
                "title": title,
                "description": description,
                "type": "github",
                "branch": "main",
                "repo": "tanstack/table",
                "subPath": sub_path,
            })
        })
        .collect();

    // Await all the futures

    futures::future::join_all(futures).await
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}
