use crate::locales::Locale;
use anyhow::Result;
use include_dir::{include_dir, Dir};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::io::Read;
use std::sync::Mutex;

pub type StopWords = Vec<String>;

static STOP_WORDS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/src/stop_words");

static STOP_WORDS_CACHE: Lazy<Mutex<HashMap<String, Option<StopWords>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn get_stop_words(locale: Locale) -> Result<Option<StopWords>> {
    let locale_as_str = locale.to_string().to_string();

    let mut cache = STOP_WORDS_CACHE.lock().unwrap();
    if let Some(cached) = cache.get(&locale_as_str) {
        return Ok(cached.clone());
    }

    let file = STOP_WORDS_DIR.get_file(format!("{}.txt", locale.to_string()));
    let stop_words = match file {
        Some(file) => {
            let contents = file.contents_utf8().unwrap_or_default();
            let words: StopWords = contents.lines().map(|line| line.to_string()).collect();
            Some(words)
        }
        None => None,
    };

    cache.insert(locale_as_str, stop_words.clone());
    Ok(stop_words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::locales::Locale;

    #[test]
    fn test_get_en_stop_words() {
        let stop_words = get_stop_words(Locale::EN).unwrap().unwrap();
        assert_eq!(stop_words.contains(&"each".to_string()), true);
    }

    #[test]
    fn test_get_it_stop_words() {
        let stop_words = get_stop_words(Locale::IT).unwrap().unwrap();
        assert_eq!(stop_words.contains(&"abbiamo".to_string()), true);
    }
}
