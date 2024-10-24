use smallvec::SmallVec;


struct EncodedKey(SmallVec<[u8; 64]>);

struct Node<K, V> {
    key: EncodedKey,
    key_value: Option<Box<(K, V)>>,
    children: [Option<Box<Node<K, V>>>; 16],
}

impl<K: AsKey, V> Node<K, V> {
    fn insert(&self, encoded_key: EncodedKey, key: K, value: V) {
        iterative_insert(self, encoded_key, key, value);
    }
}

fn iterative_insert<K: AsKey, V>(root: &Node<K, V>, encoded_key: EncodedKey, key: K, value: V) {
    if encoded_key.0.len() == 0 {
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
                    depth += child.key.len();
                    prev = child;
                }
            }
        } else {
            let node_key = encoded_key.0.split_at(depth);
            current.add_child(
                bucket,
                Box::new(Node::with_key_value(node_key, key, value)),
            );
            return None;
        }
    }
}

struct RadixTree<K, V> {
    node: Node<K, V>,
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
