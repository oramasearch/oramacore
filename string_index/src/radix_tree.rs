use std::sync::RwLock;



#[derive(Debug)]
pub struct Tree<K: AsTreeKey, V> {
    /// The number of values stored in this sub-trie (this node and all descendants).
    length: usize,
    /// The main content of this trie.
    root: TreeNode<K, V>,
}

const BRANCH_FACTOR: usize = 16;

#[derive(Debug)]
struct TreePair<K: AsTreeKey, V> {
    key: K,
    value: V,
}

#[derive(Debug)]
struct TreeNode<K: AsTreeKey, V> {
    /// Key fragments/bits associated with this node, such that joining the keys from all
    /// parent nodes and this node is equal to the bit-encoding of this node's key.
    pub key: Vec<u8>,

    /// The key and value stored at this node.
    pub key_value: Option<TreePair<K, V>>,

    /// The number of children which are Some rather than None.
    pub child_count: usize,

    /// The children of this node stored such that the first nibble of each child key
    /// dictates the child's bucket.
    pub children: [Option<Box<RwLock<TreeNode<K, V>>>>; BRANCH_FACTOR],
}

impl<K: AsTreeKey, V> TreeNode<K, V> {
    fn new() -> Self {
        TreeNode {
            key: Vec::new(),
            key_value: None,
            children: Default::default(),
            child_count: 0,
        }
    }

    fn with_key_value(key: Vec<u8>, pair: TreePair<K, V>) -> Self {
        TreeNode {
            key,
            key_value: Some(pair),
            children: Default::default(),
            child_count: 0,
        }

    }
}

pub trait AsTreeKey {
    fn as_tree_key(&self) -> Vec<u8>;
}

impl<K: AsTreeKey, V> Tree<K, V> {
    pub fn new() -> Tree<K, V> {
        Tree {
            length: 0,
            root: TreeNode::new(),
        }
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let encoded_key = key.as_tree_key();
        let pair = TreePair { key, value };
        let result = iterative_insert(self.root, encoded_key, pair);
        if result.is_none() {
            self.length += 1;
        }
        result
    }
}


#[inline]
fn iterative_insert<K: AsTreeKey, V>(root: &mut TreeNode<K, V>, encoded_key: Vec<u8>, pair: TreePair<K, V>) -> Option<V>
{
    if encoded_key.len() == 0 {
        // TODO: Handle this case.
        panic!("Key is empty");
    }

    let mut prev = trie;
    let mut depth = 0;

    loop {
        let bucket = nv.get(depth) as usize;
        let current = prev;
        if let Some(ref mut child) = current.children[bucket] {
            match match_keys(depth, &nv, &child.key) {
                KeyMatch::Full => {
                    return child.replace_value(key, value);
                }
                KeyMatch::Partial(idx) => {
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
                KeyMatch::FirstPrefix => {
                    child.split(nv.len() - depth);
                    child.add_key_value(key, value);
                    return None;
                }
                KeyMatch::SecondPrefix => {
                    depth += child.key.len();
                    prev = child;
                }
            }
        } else {
            let node_key = nv.split(depth);
            current.add_child(
                bucket,
                Box::new(TreeNode::with_key_value(node_key, key, value)),
            );
            return None;
        }
    }
}
