use std::fs;

use serde_json::Value;

use crate::fs_utils::get_files;


pub fn parse_example(path: &str) -> Vec<Value> {
    let all_files = get_files(path.parse().unwrap(), vec!["tsx".to_string()]);

    let mut examples = Vec::with_capacity(all_files.len());
    for file in all_files {
        let file_path = file.path();
        let content = fs::read_to_string(file_path.clone()).unwrap();

        let example = serde_json::json!({
            "type": "example",
            "id": file_path.to_string_lossy().to_string(),
            "code": content,
        });
        examples.push(example);
    }

    examples
}
