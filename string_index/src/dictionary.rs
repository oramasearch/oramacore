use std::sync::atomic::AtomicUsize;

use dashmap::DashMap;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TermId(pub usize);

#[derive(Debug)]
pub struct Dictionary {
    // This is hashmap.
    // To perform better, we should use to a radix tree.
    // TODO: move to radix tree when we have a concurrent radix tree.
    index: DashMap<String, TermId>,
    max_term_id: AtomicUsize,
}

impl Dictionary {
    pub fn new() -> Self {
        Dictionary {
            index: DashMap::new(),
            max_term_id: AtomicUsize::new(0),
        }
    }

    pub fn get_or_add(&self, term: &str) -> TermId {
        let output = self.index.entry(term.to_string()).or_insert_with(|| {
            TermId(
                self.max_term_id
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel),
            )
        });

        *output
    }
}
