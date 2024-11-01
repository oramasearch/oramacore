use std::collections::HashMap;

fn main() {
    let stop_words: HashMap<_, _> = std::fs::read_dir("./src/stop_words")
        .unwrap()
        .filter_map(|dir| dir.ok())
        .filter_map(|dir| {
            let filename = dir.file_name().to_str()?.to_string();
            let stop_words = std::fs::read_to_string(dir.path()).ok()?;

            Some((filename, stop_words))
        })
        .collect();
}
