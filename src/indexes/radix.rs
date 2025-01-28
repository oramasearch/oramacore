use anyhow::Result;
use ptrie::Trie;

#[derive(Debug)]
pub struct RadixIndex<Value> {
    pub inner: Trie<u8, Value>,
}

impl<Value: Clone> RadixIndex<Value> {
    pub fn new() -> Self {
        Self { inner: Trie::new() }
    }

    pub fn get_mut<I: Iterator<Item = u8>>(&mut self, key: I) -> Option<&mut Value> {
        self.inner.get_mut(key)
    }

    pub fn insert<I: Iterator<Item = u8>>(&mut self, key: I, value: Value) {
        self.inner.insert(key, value);
    }

    pub fn search<'s, 'input>(&'s self, token: &'input str) -> Result<Vec<&'s Value>>
    where
        'input: 's,
    {
        let (exact_match, mut others) = self.inner.find_postfixes_with_current(token.bytes());

        if let Some(value) = exact_match {
            others.push(value);
        }

        Ok(others)
    }
}
