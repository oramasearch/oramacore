use crate::locales::Locale;
use anyhow::Result;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

pub type StopWords = Vec<String>;

pub fn get_stop_words(locale: Locale) -> Result<Option<StopWords>, io::Error> {
    let stop_words_dir = Path::new("nlp/src/stop_words");
    let file_path = stop_words_dir.join(format!("{}.txt", locale.to_str()));

    let mut file = match File::open(&file_path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };

    let mut stop_words = String::new();
    file.read_to_string(&mut stop_words)?;

    let words: StopWords = stop_words.lines().map(|line| line.to_string()).collect();

    Ok(Some(words))
}
