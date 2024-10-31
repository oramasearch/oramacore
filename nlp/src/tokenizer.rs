use std::collections::HashSet;

use crate::locales::Locale;
use crate::stop_words::get_stop_words;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    split_regex: Regex,
    stop_words: HashSet<String>,
}

impl Tokenizer {
    pub fn english() -> Self {
        let stop_words = get_stop_words(Locale::EN).unwrap().unwrap();
        Tokenizer {
            split_regex: Regex::new(r"[^A-Za-zàèéìòóù0-9_']+").unwrap(),
            stop_words: stop_words.into_iter().collect(),
        }
    }

    pub fn tokenize<'a, 'b>(&'a self, input: &'b str) -> impl Iterator<Item = String> + 'b
    where
        'a: 'b,
    {
        let b = self
            .split_regex
            .split(input)
            .filter(|token| !token.is_empty())
            .filter_map(|token| self.normalize_token(token.to_lowercase()))
            .filter(|token| !token.is_empty() && !self.stop_words.contains(token.as_str()));
        b
    }

    fn normalize_token(&self, token: String) -> Option<String> {
        if self.stop_words.contains(token.as_str()) {
            return None;
        }
        Some(self.replace_diacritics(token.as_str()))
    }

    fn replace_diacritics(&self, str: &str) -> String {
        str.chars().map(replace_char).collect()
    }
}

fn replace_char(c: char) -> char {
    let code = u32::from(c);
    if !(DIACRITICS_CHARCODE_START..=DIACRITICS_CHARCODE_END).contains(&code) {
        return c;
    }

    let index = usize::try_from(code - DIACRITICS_CHARCODE_START)
        .expect("Invalid index: code - DIACRITICS_CHARCODE_START is not a valid usize");
    *CHARCODE_REPLACE_MAPPING.get(index).unwrap_or(&c)
}

const CHARCODE_REPLACE_MAPPING: &[char] = &[
    'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'E', 'E', 'E', 'E', 'I', 'I', 'I', 'I', 'E', 'N', 'O',
    'O', 'O', 'O', 'O', '\0', 'O', 'U', 'U', 'U', 'U', 'Y', 'P', 's', 'a', 'a', 'a', 'a', 'a', 'a',
    'a', 'c', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i', 'e', 'n', 'o', 'o', 'o', 'o', 'o', '\0', 'o',
    'u', 'u', 'u', 'u', 'y', 'p', 'y', 'A', 'a', 'A', 'a', 'A', 'a', 'C', 'c', 'C', 'c', 'C', 'c',
    'C', 'c', 'D', 'd', 'D', 'd', 'E', 'e', 'E', 'e', 'E', 'e', 'E', 'e', 'E', 'e', 'G', 'g', 'G',
    'g', 'G', 'g', 'G', 'g', 'H', 'h', 'H', 'h', 'I', 'i', 'I', 'i', 'I', 'i', 'I', 'i', 'I', 'i',
    'I', 'i', 'J', 'j', 'K', 'k', 'k', 'L', 'l', 'L', 'l', 'L', 'l', 'L', 'l', 'L', 'l', 'N', 'n',
    'N', 'n', 'N', 'n', 'n', 'N', 'n', 'O', 'o', 'O', 'o', 'O', 'o', 'O', 'o', 'R', 'r', 'R', 'r',
    'R', 'r', 'S', 's', 'S', 's', 'S', 's', 'S', 's', 'T', 't', 'T', 't', 'T', 't', 'U', 'u', 'U',
    'u', 'U', 'u', 'U', 'u', 'U', 'u', 'U', 'u', 'W', 'w', 'Y', 'y', 'Y', 'Z', 'z', 'Z', 'z', 'Z',
    'z', 's',
];

const DIACRITICS_CHARCODE_START: u32 = 192;
const DIACRITICS_CHARCODE_END: u32 = 383;

static STOP_WORDS_ENGLISH: &[&str] = &[
    "a", // <---- this is added in rust code
    "i",
    "me",
    "my",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
    "ought",
    "i'm",
    "you're",
    "he's",
    "she's",
    "it's",
    "we're",
    "they're",
    "i've",
    "you've",
    "we've",
    "they've",
    "i'd",
    "you'd",
    "he'd",
    "she'd",
    "we'd",
    "they'd",
    "i'll",
    "you'll",
    "he'll",
    "she'll",
    "we'll",
    "they'll",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    "hadn't",
    "doesn't",
    "don't",
    "didn't",
    "won't",
    "wouldn't",
    "shan't",
    "shouldn't",
    "can't",
    "cannot",
    "couldn't",
    "mustn't",
    "let's",
    "that's",
    "who's",
    "what's",
    "here's",
    "there's",
    "when's",
    "where's",
    "why's",
    "how's",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
];

#[cfg(test)]
mod tests {
    #[test]
    fn test_tokenizer() {
        let tokenizer = super::Tokenizer::english();
        let tokens: Vec<String> = tokenizer.tokenize("Hello, world!").collect();
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenizer_2() {
        let tokenizer = super::Tokenizer::english();
        let tokens: Vec<String> = tokenizer.tokenize("Hello, - world!").collect();
        assert_eq!(tokens, vec!["hello", "world"]);
    }
}
