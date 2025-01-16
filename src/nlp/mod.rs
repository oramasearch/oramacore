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

use crate::types::StringParser;

pub struct TextParser {
    locale: Locale,
    tokenizer: Tokenizer,
    stemmer: Stemmer,
}

impl Debug for TextParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextParser")
            .field("locale", &self.locale)
            .field("tokenizer", &"...".to_string())
            .field("stemmer", &"...".to_string())
            .finish()
    }
}
impl TextParser {
    pub fn from_locale(locale: Locale) -> Self {
        let (tokenizer, stemmer) = match locale {
            Locale::AR => (Tokenizer::arab(), Stemmer::create(Algorithm::Arabic)),
            // Locale::BG => (Tokenizer::bulgarian(), Stemmer::create(Algorithm::Bulgarian)), @todo: support bulgarian stemming
            Locale::DA => (Tokenizer::danish(), Stemmer::create(Algorithm::Danish)),
            Locale::DE => (Tokenizer::german(), Stemmer::create(Algorithm::German)),
            Locale::EN => (Tokenizer::english(), Stemmer::create(Algorithm::English)),
            Locale::EL => (Tokenizer::greek(), Stemmer::create(Algorithm::Greek)),
            Locale::ES => (Tokenizer::spanish(), Stemmer::create(Algorithm::Spanish)),
            // Locale::ET => (Tokenizer::estonian(), Stemmer::create(Algorithm::Estonian)), @todo: support estonian stemmer
            Locale::FI => (Tokenizer::finnish(), Stemmer::create(Algorithm::Finnish)),
            Locale::FR => (Tokenizer::french(), Stemmer::create(Algorithm::French)),
            // Locale::GA => (Tokenizer::irish(), Stemmer::create(Algorithm::Irish)), @todo: support irish stemmer
            // Locale::HI => (Tokenizer::hindi(), Stemmer::create(Algorithm::Hindi)), @todo: support hindi stemmer
            Locale::HU => (
                Tokenizer::hungarian(),
                Stemmer::create(Algorithm::Hungarian),
            ),
            // Locale::HY => (Tokenizer::armenian(), Stemmer::create(Algorithm::Armenian)), @todo: support armenian stemmer
            // Locale::ID => (Tokenizer::indonesian(), Stemmer::create(Algorithm::Indonesian)), @todo: support indonesian stemmer
            Locale::IT => (Tokenizer::italian(), Stemmer::create(Algorithm::Italian)),
            // Locale::JP => (Tokenizer::japanese(), Stemmer::create(Algorithm::Japanese)), @todo: support japanese stemmer
            // Locale::KO => (Tokenizer::korean(), Stemmer::create(Algorithm::Korean)), @todo: support korean stemmer
            // Locale::LT => (Tokenizer::lithuanian(), Stemmer::create(Algorithm::Lithuanian)) @todo: support lithuanian stemmer
            // Locale::NE => (Tokenizer::nepali(), Stemmer::create(Algorithm::Nepali)) @todo: support nepali stemmer
            Locale::NL => (Tokenizer::dutch(), Stemmer::create(Algorithm::Dutch)),
            Locale::NO => (
                Tokenizer::norwegian(),
                Stemmer::create(Algorithm::Norwegian),
            ),
            Locale::PT => (
                Tokenizer::portuguese(),
                Stemmer::create(Algorithm::Portuguese),
            ),
            Locale::RO => (Tokenizer::romanian(), Stemmer::create(Algorithm::Romanian)),
            Locale::RU => (Tokenizer::russian(), Stemmer::create(Algorithm::Russian)),
            // Locale::SA => (Tokenizer::sanskrit(), Stemmer::create(Algorithm::Sanskrit)) @todo: support sanskrit stemmer
            // Locale::SL => (Tokenizer::slovenian(), Stemmer::create(Algorithm::Slovenian)) @todo: support slovenian stemmer
            // Locale::SR => (Tokenizer::serbian(), Stemmer::create(Algorithm::Serbian)) @todo: support serbian stemmer
            Locale::SV => (Tokenizer::swedish(), Stemmer::create(Algorithm::Swedish)),
            Locale::TA => (Tokenizer::tamil(), Stemmer::create(Algorithm::Tamil)),
            Locale::TR => (Tokenizer::turkish(), Stemmer::create(Algorithm::Turkish)),
            // Locale::UK => (Tokenizer::ukrainian(), Stemmer::create(Algorithm::Ukrainian)) @todo: support ukrainian stemmer
            // Locale::ZH => (Tokenizer::chinese(), Stemmer::create(Algorithm::Chinese)), @todo: support chinese stemmer
            _ => (Tokenizer::english(), Stemmer::create(Algorithm::English)),
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
            .map(move |token| {
                let stemmed = self.stemmer.stem(&token).to_string();
                if stemmed == token {
                    return (token, vec![]);
                }
                (token, vec![stemmed])
            })
            .collect()
    }
}

impl StringParser for TextParser {
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
}
