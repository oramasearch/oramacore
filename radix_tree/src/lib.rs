use atom_box::{
    domain::{Domain, ReclaimStrategy},
    AtomBox,
};
use smallvec::SmallVec;
use std::sync::atomic::AtomicBool;

struct EncodedKey(SmallVec<[u8; 64]>);

const CUSTOM_DOMAIN_ID: usize = 42;
static CUSTOM_DOMAIN: Domain<CUSTOM_DOMAIN_ID> = Domain::new(ReclaimStrategy::Eager);

struct Node<'a, K, V> {
    key: EncodedKey,
    key_value: Option<AtomBox<'a, i32, CUSTOM_DOMAIN_ID>>,
    children: [Option<Box<Node<'a, K, V>>>; 16],
}

impl<'a, K: AsKey, V> Node<'a, K, V> {
    fn new() -> Self {
        let b = AtomBox::new_with_domain(5, &CUSTOM_DOMAIN);
        Self {
            v: b,
            key: EncodedKey(SmallVec::new()),
            key_value: None,
            children: Default::default(),
        }
    }

    fn insert(&self, encoded_key: EncodedKey, key: K, value: V) {
        iterative_insert(self, encoded_key, key, value);
    }

    fn replace_value(&self, key: K, value: V) {
        self.key_value = Some(Box::new((key, value)));
    }
}

fn iterative_insert<'a, K: AsKey, V>(
    root: &Node<'a, K, V>,
    encoded_key: EncodedKey,
    key: K,
    value: V,
) {
    if encoded_key.0.len() == 0 {
        // TODO: decide what to do here
        panic!("Key is empty");
    }

    let mut prev = root;
    let mut depth = 0;

    loop {
        let bucket = *encoded_key.0.get(depth).unwrap() as usize;
        let current = prev;
        if let Some(ref mut child) = current.children[bucket] {
            match match_keys(depth, &encoded_key, &child.key) {
                EncodedKeyMatch::Full => {
                    return child.replace_value(key, value);
                }
                EncodedKeyMatch::Partial(idx) => {
                    // Split the existing child.
                    child.split(idx);

                    // Insert the new key below the prefix node.
                    let new_key = nv.split(depth + idx);
                    let new_key_bucket = new_key.get(0) as usize;

                    child.add_child(
                        new_key_bucket,
                        Box::new(TrieNode::with_key_value(new_key, key, value)),
                    );

                    return None;
                }
                EncodedKeyMatch::FirstPrefix => {
                    child.split(nv.len() - depth);
                    child.add_key_value(key, value);
                    return None;
                }
                EncodedKeyMatch::SecondPrefix => {
                    depth += child.key.0.len();
                    prev = child;
                }
            }
        } else {
            let node_key = encoded_key.0.split_at(depth);
            current.add_child(bucket, Box::new(Node::with_key_value(node_key, key, value)));
            return None;
        }
    }
}

struct RadixTree<K, V> {
    node: Node<'static, K, V>,
}

impl<K: AsKey, V> RadixTree<K, V> {
    fn insert(&self, key: K, value: V) {
        let encoded_key = key.encode();
        self.node.insert(encoded_key, key, value);
    }
}

trait AsKey {
    fn encode(&self) -> EncodedKey;
}

#[inline]
pub fn match_keys(start_idx: usize, first: &EncodedKey, second: &EncodedKey) -> EncodedKeyMatch {
    let first_len = first.0.len() - start_idx;
    let second_len = second.0.len();
    let min_length = ::std::cmp::min(first_len, second_len);

    for i in 0..min_length {
        if first.0.get(start_idx + i).unwrap() != second.0.get(i).unwrap() {
            return EncodedKeyMatch::Partial(i);
        }
    }

    match (first_len, second_len) {
        (x, y) if x < y => EncodedKeyMatch::FirstPrefix,
        (x, y) if x == y => EncodedKeyMatch::Full,
        _ => EncodedKeyMatch::SecondPrefix,
    }
}

/// Key comparison result.
#[derive(Debug)]
pub enum EncodedKeyMatch {
    Partial(usize),
    FirstPrefix,
    SecondPrefix,
    Full,
}
