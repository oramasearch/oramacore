pub mod chunker;
pub mod locales;
pub mod stop_words;
pub mod tokenizer;

use std::{
    fmt::{Debug, Formatter},
    sync::Arc,
};

use anyhow::Result;
use dashmap::DashMap;
use locales::Locale;
use rust_stemmers::Algorithm;
pub use rust_stemmers::Stemmer;
use tokenizer::Tokenizer;

pub struct TextParser {
    locale: Locale,
    tokenizer: Tokenizer,
    stemmer: Option<Stemmer>,
}

impl Debug for TextParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextParser")
            .field("locale", &self.locale)
            .finish()
    }
}
impl TextParser {
    pub fn from_locale(locale: Locale) -> Self {
        let (tokenizer, stemmer) = match locale {
            Locale::AR => (Tokenizer::arab(), Some(Stemmer::create(Algorithm::Arabic))),
            Locale::BG => (Tokenizer::bulgarian(), None),
            Locale::DA => (
                Tokenizer::danish(),
                Some(Stemmer::create(Algorithm::Danish)),
            ),
            Locale::DE => (
                Tokenizer::german(),
                Some(Stemmer::create(Algorithm::German)),
            ),
            Locale::EN => (
                Tokenizer::english(),
                Some(Stemmer::create(Algorithm::English)),
            ),
            Locale::EL => (Tokenizer::greek(), Some(Stemmer::create(Algorithm::Greek))),
            Locale::ES => (
                Tokenizer::spanish(),
                Some(Stemmer::create(Algorithm::Spanish)),
            ),
            Locale::ET => (Tokenizer::estonian(), None),
            Locale::FI => (Tokenizer::finnish(), None),
            Locale::FR => (
                Tokenizer::french(),
                Some(Stemmer::create(Algorithm::French)),
            ),
            Locale::GA => (Tokenizer::irish(), None),
            Locale::HI => (Tokenizer::hindi(), None),
            Locale::HU => (
                Tokenizer::hungarian(),
                Some(Stemmer::create(Algorithm::Hungarian)),
            ),
            Locale::HY => (Tokenizer::armenian(), None),
            Locale::ID => (Tokenizer::indonesian(), None),
            Locale::IT => (
                Tokenizer::italian(),
                Some(Stemmer::create(Algorithm::Italian)),
            ),
            Locale::JP => (Tokenizer::japanese(), None),
            Locale::KO => (Tokenizer::korean(), None),
            Locale::LT => (Tokenizer::lithuanian(), None),
            Locale::NE => (Tokenizer::nepali(), None),
            Locale::NL => (Tokenizer::dutch(), Some(Stemmer::create(Algorithm::Dutch))),
            Locale::NO => (
                Tokenizer::norwegian(),
                Some(Stemmer::create(Algorithm::Norwegian)),
            ),
            Locale::PT => (
                Tokenizer::portuguese(),
                Some(Stemmer::create(Algorithm::Portuguese)),
            ),
            Locale::RO => (
                Tokenizer::romanian(),
                Some(Stemmer::create(Algorithm::Romanian)),
            ),
            Locale::RU => (
                Tokenizer::russian(),
                Some(Stemmer::create(Algorithm::Russian)),
            ),
            Locale::SA => (Tokenizer::sanskrit(), None),
            Locale::SL => (Tokenizer::slovenian(), None),
            Locale::SR => (Tokenizer::serbian(), None),
            Locale::SV => (
                Tokenizer::swedish(),
                Some(Stemmer::create(Algorithm::Swedish)),
            ),
            Locale::TA => (Tokenizer::tamil(), Some(Stemmer::create(Algorithm::Tamil))),
            Locale::TR => (
                Tokenizer::turkish(),
                Some(Stemmer::create(Algorithm::Turkish)),
            ),
            Locale::UK => (Tokenizer::ukrainian(), None),
            Locale::ZH => (Tokenizer::chinese(), None),
        };
        Self {
            locale,
            tokenizer,
            stemmer,
        }
    }

    pub fn locale(&self) -> Locale {
        self.locale
    }

    pub fn tokenize(&self, input: &str) -> Vec<String> {
        self.tokenizer.tokenize(input).collect()
    }

    pub fn tokenize_and_stem(&self, input: &str) -> Vec<(String, Vec<String>)> {
        self.tokenizer
            .tokenize(input)
            .map(move |token| match &self.stemmer {
                Some(stemmer) => {
                    let stemmed = stemmer.stem(&token).to_string();
                    if stemmed == token {
                        return (token, vec![]);
                    }
                    (token, vec![stemmed])
                }
                None => (token, vec![]),
            })
            .collect()
    }

    fn tokenize_str_and_stem(&self, input: &str) -> Result<Vec<(String, Vec<String>)>> {
        Ok(self.tokenize_and_stem(input))
    }
}

#[derive(Debug)]
pub struct NLPService {
    parser: DashMap<Locale, Arc<TextParser>>,
}
impl Default for NLPService {
    fn default() -> Self {
        Self::new()
    }
}

impl NLPService {
    pub fn new() -> Self {
        Self {
            parser: Default::default(),
        }
    }

    pub fn get(&self, locale: Locale) -> Arc<TextParser> {
        match self.parser.entry(locale) {
            dashmap::Entry::Occupied(occupied_entry) => occupied_entry.get().clone(),
            dashmap::Entry::Vacant(vacant_entry) => {
                let parser = TextParser::from_locale(locale);
                let parser = Arc::new(parser);
                vacant_entry.insert(parser.clone());
                parser
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let parser = TextParser::from_locale(Locale::EN);

        let output = parser.tokenize("Hello, world!");
        assert_eq!(output, vec!["hello", "world"]);

        let output = parser.tokenize("Hello, world! fruitlessly");
        assert_eq!(output, vec!["hello", "world", "fruitlessly"]);
    }

    #[test]
    fn test_tokenize_and_stem() {
        let parser = TextParser::from_locale(Locale::EN);

        let output = parser.tokenize_and_stem("Hello, world!");
        assert_eq!(
            output,
            vec![("hello".to_string(), vec![]), ("world".to_string(), vec![])]
        );

        let output = parser.tokenize_and_stem("Hello, world! fruitlessly");
        assert_eq!(
            output,
            vec![
                ("hello".to_string(), vec![]),
                ("world".to_string(), vec![]),
                ("fruitlessly".to_string(), vec!["fruitless".to_string()])
            ]
        );
    }

    #[test]
    fn test_lang_it() {
        let parser = TextParser::from_locale(Locale::IT);

        let t = "avvocato";
        let output = parser.tokenize_and_stem(t);
        assert_eq!(
            output,
            vec![("avvocato".to_string(), vec!["avvoc".to_string()])]
        );

        let t = "avvocata";
        let output = parser.tokenize_and_stem(t);
        assert_eq!(
            output,
            vec![("avvocata".to_string(), vec!["avvoc".to_string()])]
        );
    }
}
