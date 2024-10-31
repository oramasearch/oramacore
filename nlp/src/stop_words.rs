use crate::locales::Locale;
use anyhow::Result;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub type StopWords = Vec<String>;

static STOP_WORDS_CACHE: Lazy<Mutex<HashMap<String, Option<StopWords>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn get_stop_words(locale: Locale) -> Result<Option<StopWords>, io::Error> {
    let locale_as_str = locale.to_str().to_string();

    let mut cache = STOP_WORDS_CACHE.lock().unwrap();
    if let Some(cached) = cache.get(&locale_as_str) {
        return Ok(cached.clone());
    }

    let stop_words_dir = stop_words_path();
    let file_path = stop_words_dir.join(format!("{}.txt", locale.to_str()));

    let mut file = match File::open(&file_path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            cache.insert(locale_as_str, None);
            return Ok(None);
        }
        Err(e) => return Err(e),
    };

    let mut stop_words = String::new();
    file.read_to_string(&mut stop_words)?;

    let words: StopWords = stop_words.lines().map(|line| line.to_string()).collect();

    cache.insert(locale_as_str, Some(words.clone()));
    Ok(Some(words))
}

fn stop_words_path() -> PathBuf {
    let base_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    Path::new(base_dir).join("src/stop_words")
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
