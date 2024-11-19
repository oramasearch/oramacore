use crate::nlp::locales::Locale;

include!(concat!(
    env!("OUT_DIR"),
    "/nlp/stop_words/stop_words_gen/mod.rs"
));

pub fn get_stop_words(locale: Locale) -> Option<StopWords> {
    let hash_map = STOP_WORDS_CACHE.get_or_init(load_stop_words);
    hash_map.get(&locale).cloned()?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlp::locales::Locale;

    #[test]
    fn test_stop_words1() {
        let stop_words = get_stop_words(Locale::EN).unwrap();
        assert!(stop_words.contains(&"each"));
    }
    #[test]
    fn test_stop_words2() {
        let stop_words = get_stop_words(Locale::EN).unwrap();
        assert!(stop_words.contains(&"each"));
    }

    #[test]
    fn test_get_en_stop_words() {
        let stop_words = get_stop_words(Locale::EN).unwrap();
        assert!(stop_words.contains(&"each"));
    }

    #[test]
    fn test_get_it_stop_words() {
        let stop_words = get_stop_words(Locale::IT).unwrap();
        assert!(stop_words.contains(&"abbiamo"));
    }
}
