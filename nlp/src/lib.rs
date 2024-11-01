pub mod chunker;
pub mod locales;
pub mod stop_words;
pub mod tokenizer;

use rust_stemmers::Algorithm;
pub use rust_stemmers::Stemmer;
use tokenizer::Tokenizer;
use crate::locales::Locale;

pub struct Parser {
    pub tokenizer: Tokenizer,
    stemmer: Stemmer,
}
impl Parser {
    pub fn from_language(locale: Locale) -> Self {
        let (tokenizer, stemmer) = match locale {
            Locale::EN => (Tokenizer::english(), Stemmer::create(Algorithm::English)),
            // @todo: manage other locales
            _ => (Tokenizer::english(), Stemmer::create(Algorithm::English)),
        };
        Self { tokenizer, stemmer }
    }
}

pub fn tokenize<'a, 'b>(
    input: &'b str,
    tokenizer: &'a Tokenizer,
) -> impl Iterator<Item = String> + 'b
where
    'a: 'b,
{
    tokenizer.tokenize(input)
}

pub fn tokenize_and_stem<'a, 'b>(
    input: &'b str,
    parser: &'a Parser,
) -> impl Iterator<Item = (String, String)> + 'b
where
    'a: 'b,
{
    parser.tokenizer.tokenize(input).map(move |token| {
        let stemmed = parser.stemmer.stem(&token).to_string();
        (token, stemmed)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokenizer = Tokenizer::english();

        let output = tokenize("Hello, world!", &tokenizer).collect::<Vec<String>>();
        assert_eq!(output, vec!["hello", "world"]);

        let output = tokenize("Hello, world! fruitlessly", &tokenizer).collect::<Vec<String>>();
        assert_eq!(output, vec!["hello", "world", "fruitlessly"]);
    }

    #[test]
    fn test_tokenize_and_stem() {
        let parser = Parser::from_language(Locale::EN);

        let output = tokenize_and_stem("Hello, world!", &parser).collect::<Vec<(String, String)>>();
        assert_eq!(
            output,
            vec![
                ("hello".to_string(), "hello".to_string()),
                ("world".to_string(), "world".to_string())
            ]
        );

        let output = tokenize_and_stem("Hello, world! fruitlessly", &parser)
            .collect::<Vec<(String, String)>>();
        assert_eq!(
            output,
            vec![
                ("hello".to_string(), "hello".to_string()),
                ("world".to_string(), "world".to_string()),
                ("fruitlessly".to_string(), "fruitless".to_string())
            ]
        );
    }
}
