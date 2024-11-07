use serde_json::Value;
use types::Document;

pub fn get_embeddable_string(document: &Document) -> String {
    let mut embeddable_string = String::new();

    for value in document.inner.values() {
        if let Value::String(s) = value {
            embeddable_string.push_str(s);
        }
    }

    embeddable_string
}
