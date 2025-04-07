use std::collections::HashSet;

use crate::locales::Locale;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    split_regex: Regex,
    stop_words: HashSet<&'static str>,
}

impl Tokenizer {
    fn new(locale: Locale) -> Self {
        let stop_words: HashSet<&'static str> = locale.stop_words().unwrap();
        Tokenizer {
            split_regex: locale.split_regex().unwrap(),
            stop_words,
        }
    }

    pub fn arab() -> Self {
        Self::new(Locale::AR)
    }

    pub fn bulgarian() -> Self {
        Self::new(Locale::BG)
    }

    pub fn danish() -> Self {
        Self::new(Locale::DA)
    }

    pub fn german() -> Self {
        Self::new(Locale::DE)
    }

    pub fn greek() -> Self {
        Self::new(Locale::EL)
    }

    pub fn english() -> Self {
        Self::new(Locale::EN)
    }

    pub fn spanish() -> Self {
        Self::new(Locale::ES)
    }

    pub fn estonian() -> Self {
        Self::new(Locale::ET)
    }

    pub fn finnish() -> Self {
        Self::new(Locale::FI)
    }

    pub fn french() -> Self {
        Self::new(Locale::FR)
    }

    pub fn irish() -> Self {
        Self::new(Locale::GA)
    }

    pub fn hindi() -> Self {
        Self::new(Locale::HI)
    }

    pub fn hungarian() -> Self {
        Self::new(Locale::HU)
    }

    pub fn armenian() -> Self {
        Self::new(Locale::HY)
    }

    pub fn indonesian() -> Self {
        Self::new(Locale::ID)
    }

    pub fn italian() -> Self {
        Self::new(Locale::IT)
    }

    pub fn japanese() -> Self {
        Self::new(Locale::JP)
    }

    pub fn korean() -> Self {
        Self::new(Locale::KO)
    }

    pub fn lithuanian() -> Self {
        Self::new(Locale::LT)
    }

    pub fn nepali() -> Self {
        Self::new(Locale::NE)
    }

    pub fn dutch() -> Self {
        Self::new(Locale::NL)
    }

    pub fn norwegian() -> Self {
        Self::new(Locale::NO)
    }

    pub fn portuguese() -> Self {
        Self::new(Locale::PT)
    }

    pub fn romanian() -> Self {
        Self::new(Locale::RO)
    }

    pub fn russian() -> Self {
        Self::new(Locale::RU)
    }

    pub fn sanskrit() -> Self {
        Self::new(Locale::SA)
    }

    pub fn slovenian() -> Self {
        Self::new(Locale::SL)
    }

    pub fn serbian() -> Self {
        Self::new(Locale::SR)
    }

    pub fn swedish() -> Self {
        Self::new(Locale::SV)
    }

    pub fn tamil() -> Self {
        Self::new(Locale::TA)
    }

    pub fn turkish() -> Self {
        Self::new(Locale::TR)
    }

    pub fn ukrainian() -> Self {
        Self::new(Locale::UK)
    }

    pub fn chinese() -> Self {
        Self::new(Locale::ZH)
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
        assert_eq!(tokens, vec!["hello", "-", "world"]);
    }
}
