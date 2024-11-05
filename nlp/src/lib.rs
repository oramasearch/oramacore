pub mod chunker;
pub mod locales;
pub mod stop_words;
pub mod tokenizer;

use crate::locales::Locale;
use rust_stemmers::Algorithm;
pub use rust_stemmers::Stemmer;
use tokenizer::Tokenizer;

pub struct TextParser {
    locale: Locale,
    pub tokenizer: Tokenizer,
    stemmer: Stemmer,
}

impl Clone for TextParser {
    fn clone(&self) -> Self {
        let stemmer = match self.locale {
            Locale::EN => Stemmer::create(Algorithm::English),
            // @todo: manage other locales
            _ => Stemmer::create(Algorithm::English),
        };
        Self {
            locale: self.locale,
            tokenizer: self.tokenizer.clone(),
            stemmer,
        }
    }
}

impl TextParser {
    pub fn from_language(locale: Locale) -> Self {
        let (tokenizer, stemmer) = match locale {
            Locale::EN => (Tokenizer::english(), Stemmer::create(Algorithm::English)),
            // @todo: manage other locales
            _ => (Tokenizer::english(), Stemmer::create(Algorithm::English)),
        };
        Self {
            locale,
            tokenizer,
            stemmer,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let parser = TextParser::from_language(Locale::EN);

        let output = parser.tokenize("Hello, world!");
        assert_eq!(output, vec!["hello", "world"]);

        let output = parser.tokenize("Hello, world! fruitlessly");
        assert_eq!(output, vec!["hello", "world", "fruitlessly"]);
    }

    #[test]
    fn test_tokenize_and_stem() {
        let parser = TextParser::from_language(Locale::EN);

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
