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
    // This is the reverse of index. This is bad.
    // We should remove this.
    reverse_index: DashMap<TermId, String>,
}

impl Dictionary {
    pub fn new() -> Self {
        Dictionary {
            index: DashMap::new(),
            max_term_id: AtomicUsize::new(0),
            reverse_index: DashMap::new(),
        }
    }

    pub fn get_or_add(&self, term: &str) -> TermId {
        let output = self.index.entry(term.to_string()).or_insert_with(|| {
            TermId(
                self.max_term_id
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel),
            )
        });
        self.reverse_index
            .entry(*output)
            .or_insert(term.to_string());

        *output
    }

    // This is a bad function.
    // TODO: Remove me
    pub fn retrive(&self, term_id: TermId) -> String {
        let output = self.reverse_index.get(&term_id);

        let a = output.unwrap();

        (*a).clone()
    }
}
