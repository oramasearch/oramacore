use bloomfilter::Bloom;
use dashmap::DashMap;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use string_index::Posting;

const MAX_TRANSITIONS: usize = 32;
const CACHE_LINE: usize = 64;
const REBUILD_THRESHOLD: f64 = 0.3;
const BLOOM_ERROR_RATE: f64 = 0.01;

#[repr(C, align(64))]
#[derive(Clone)]
struct TransitionBlock {
    count: u16,
    _pad1: [u8; 2],
    chars: [u8; MAX_TRANSITIONS],
    _pad2: [u8; 28],
    next_states: [u32; MAX_TRANSITIONS],
    posting_ids: [u32; MAX_TRANSITIONS],
}

impl TransitionBlock {
    #[inline(always)]
    fn new() -> Self {
        Self {
            count: 0,
            _pad1: [0; 2],
            chars: [0; MAX_TRANSITIONS],
            _pad2: [0; 28],
            next_states: [0; MAX_TRANSITIONS],
            posting_ids: [0; MAX_TRANSITIONS],
        }
    }

    #[inline(always)]
    fn find_char(&self, b: u8) -> Option<usize> {
        if self.count <= 4 {
            if self.count >= 1 && self.chars[0] == b {
                return Some(0);
            }
            if self.count >= 2 && self.chars[1] == b {
                return Some(1);
            }
            if self.count >= 3 && self.chars[2] == b {
                return Some(2);
            }
            if self.count >= 4 && self.chars[3] == b {
                return Some(3);
            }
            return None;
        }
        for i in 0..self.count as usize {
            if self.chars[i] == b {
                return Some(i);
            }
        }
        None
    }

    #[inline(always)]
    fn add_transition(&mut self, b: u8, next_state: u32) -> usize {
        debug_assert!(self.count < MAX_TRANSITIONS as u16);
        let pos = self.count as usize;
        self.chars[pos] = b;
        self.next_states[pos] = next_state;
        self.count += 1;
        pos
    }
}

pub struct FST {
    transitions: Vec<TransitionBlock>,
    postings: Box<DashMap<u32, Arc<Posting>>>,
    free_blocks: Vec<u32>,
    deleted_words: Bloom<String>,
    deletion_count: usize,
    total_insertions: usize,
    last_rebuild: Instant,
}

impl FST {
    pub fn new() -> Self {
        let initial_capacity = 100_000;
        Self {
            transitions: vec![TransitionBlock::new()],
            postings: Box::new(DashMap::with_capacity_and_shard_amount(1_000_000, 32)),
            free_blocks: Vec::with_capacity(10_000),
            deleted_words: Bloom::new_for_fp_rate(initial_capacity, BLOOM_ERROR_RATE),
            deletion_count: 0,
            total_insertions: 0,
            last_rebuild: Instant::now(),
        }
    }

    pub fn delete(&mut self, word: &str) {
        self.deleted_words.set(&word.to_string());
        self.deletion_count += 1;
        if self.should_rebuild() {
            self.rebuild();
        }
    }

    #[inline]
    pub fn search(&self, word: &str) -> Option<Arc<Posting>> {
        if self.deleted_words.check(&word.to_string()) {
            return None;
        }

        let bytes = word.as_bytes();
        if bytes.is_empty() {
            return None;
        }

        if bytes.len() == 1 {
            let block = &self.transitions[0];
            let pos = block.find_char(bytes[0])?;
            return self
                .postings
                .get(&block.posting_ids[pos])
                .map(|p| Arc::clone(&p));
        }

        let mut current_block = 0;
        let last_idx = bytes.len() - 1;

        for &b in &bytes[..last_idx] {
            let block = &self.transitions[current_block];
            let pos = block.find_char(b)?;
            current_block = block.next_states[pos] as usize;
        }

        let block = &self.transitions[current_block];
        let pos = block.find_char(bytes[last_idx])?;
        self.postings
            .get(&block.posting_ids[pos])
            .map(|p| Arc::clone(&p))
    }

    #[inline]
    pub fn insert(&mut self, word: &str, posting: Posting) -> u32 {
        self.total_insertions += 1;
        if self.deleted_words.check(&word.to_string()) && self.should_rebuild() {
            self.rebuild();
        }

        let bytes = word.as_bytes();
        if bytes.is_empty() {
            return 0;
        }

        let posting_id = self.store_posting(posting);
        let mut current_block = 0;

        if bytes.len() == 1 {
            let block = &mut self.transitions[current_block];
            let pos = block
                .find_char(bytes[0])
                .unwrap_or_else(|| block.add_transition(bytes[0], 0));
            block.posting_ids[pos] = posting_id;
            return posting_id;
        }

        let last_idx = bytes.len() - 1;
        for &b in &bytes[..last_idx] {
            let block = &self.transitions[current_block];
            let (next_block, create_new) = block
                .find_char(b)
                .map(|pos| (block.next_states[pos] as usize, false))
                .unwrap_or_else(|| (self.allocate_block(), true));

            if create_new {
                let block = &mut self.transitions[current_block];
                block.add_transition(b, next_block as u32);
            }
            current_block = next_block;
        }

        let b = bytes[last_idx];
        let block = &mut self.transitions[current_block];
        let pos = block
            .find_char(b)
            .unwrap_or_else(|| block.add_transition(b, 0));
        block.posting_ids[pos] = posting_id;
        posting_id
    }

    fn should_rebuild(&self) -> bool {
        let deletion_ratio = self.deletion_count as f64 / self.total_insertions as f64;
        deletion_ratio > REBUILD_THRESHOLD
            || self.last_rebuild.elapsed() > Duration::from_secs(3600)
    }

    fn rebuild(&mut self) {
        let mut valid_entries = Vec::new();
        self.collect_valid_entries(0, String::new(), &mut valid_entries);
        *self = FST::new();
        for (word, posting) in valid_entries {
            if let Ok(posting) = Arc::try_unwrap(posting) {
                self.insert(&word, posting);
            }
        }
        self.last_rebuild = Instant::now();
    }

    fn collect_valid_entries(
        &self,
        block_idx: usize,
        prefix: String,
        entries: &mut Vec<(String, Arc<Posting>)>,
    ) {
        let block = &self.transitions[block_idx];
        for i in 0..block.count as usize {
            let mut word = prefix.clone();
            word.push(block.chars[i] as char);
            if block.posting_ids[i] != 0 {
                if let Some(posting) = self.postings.get(&block.posting_ids[i]) {
                    if !self.deleted_words.check(&word) {
                        entries.push((word.clone(), Arc::clone(&posting)));
                    }
                }
            }
            if block.next_states[i] != 0 {
                self.collect_valid_entries(block.next_states[i] as usize, word, entries);
            }
        }
    }

    #[inline(always)]
    fn allocate_block(&mut self) -> usize {
        self.free_blocks
            .pop()
            .map(|id| id as usize)
            .unwrap_or_else(|| {
                let id = self.transitions.len();
                self.transitions.push(TransitionBlock::new());
                id
            })
    }

    #[inline(always)]
    fn store_posting(&mut self, posting: Posting) -> u32 {
        let id = self.postings.len() as u32;
        self.postings.insert(id, Arc::new(posting));
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use types::{DocumentId, FieldId};

    fn create_posting(doc_id: u64, field_id: u16) -> Posting {
        Posting {
            document_id: DocumentId(doc_id),
            field_id: FieldId(field_id),
            positions: vec![0],
            term_frequency: 1.0,
            doc_length: 1,
        }
    }

    #[test]
    fn test_empty_word() {
        let mut fst = FST::new();
        let posting = create_posting(1, 1);
        assert_eq!(fst.insert("", posting), 0);
        assert!(fst.search("").is_none());
    }

    #[test]
    fn test_single_char_words() {
        let mut fst = FST::new();
        let posting = create_posting(1, 1);

        let id = fst.insert("a", posting);
        assert!(id > 0);
        assert!(fst.search("a").is_some());

        fst.insert("b", create_posting(2, 1));
        assert!(fst.search("b").is_some());
        assert!(fst.search("c").is_none());
    }

    #[test]
    fn test_duplicate_words() {
        let mut fst = FST::new();
        let posting1 = create_posting(1, 1);
        let posting2 = create_posting(2, 1);

        let id1 = fst.insert("test", posting1);
        let id2 = fst.insert("test", posting2);

        assert_ne!(id1, id2);
        assert!(fst.search("test").is_some());
    }

    #[test]
    fn test_prefixes() {
        let mut fst = FST::new();

        fst.insert("test", create_posting(1, 1));
        fst.insert("testing", create_posting(2, 1));
        fst.insert("tes", create_posting(3, 1));

        assert!(fst.search("test").is_some());
        assert!(fst.search("testing").is_some());
        assert!(fst.search("tes").is_some());
        assert!(fst.search("te").is_none());
    }

    #[test]
    fn test_delete_and_rebuild() {
        let mut fst = FST::new();

        fst.insert("word1", create_posting(1, 1));
        fst.insert("word2", create_posting(2, 1));
        fst.insert("word3", create_posting(3, 1));

        assert!(fst.search("word1").is_some());
        fst.delete("word1");
        assert!(fst.search("word1").is_none());
        assert!(fst.search("word2").is_some());

        // Force rebuild
        for _ in 0..1000 {
            fst.delete("nonexistent");
        }

        assert!(fst.search("word1").is_none());
        assert!(fst.search("word2").is_some());
        assert!(fst.search("word3").is_some());
    }

    #[test]
    fn test_unicode_words() {
        let mut fst = FST::new();

        fst.insert("测试", create_posting(1, 1));
        fst.insert("тест", create_posting(2, 1));
        fst.insert("테스트", create_posting(3, 1));

        assert!(fst.search("测试").is_some());
        assert!(fst.search("тест").is_some());
        assert!(fst.search("테스트").is_some());
        assert!(fst.search("test").is_none());
    }

    #[test]
    fn test_max_transitions() {
        let mut fst = FST::new();

        for i in 0..MAX_TRANSITIONS + 1 {
            let c = (b'a' + i as u8) as char;
            let word = c.to_string();
            if i < MAX_TRANSITIONS {
                assert!(fst.insert(&word, create_posting(i as u64, 1)) > 0);
            } else {
                assert_eq!(fst.transitions[0].count, MAX_TRANSITIONS as u16);
            }
        }
    }

    #[test]
    fn test_deleted_reinsert() {
        let mut fst = FST::new();
        let posting = create_posting(1, 1);

        fst.insert("test", posting.clone());
        assert!(fst.search("test").is_some());

        fst.delete("test");
        assert!(fst.search("test").is_none());

        fst.insert("test", posting);
        assert!(fst.search("test").is_some());
    }

    #[test]
    fn test_large_scale() {
        let mut fst = FST::new();
        let word_count = 100_000;

        for i in 0..word_count {
            let word = format!("word{}", i);
            fst.insert(&word, create_posting(i as u64, 1));
        }

        for i in 0..word_count {
            let word = format!("word{}", i);
            assert!(fst.search(&word).is_some());
        }
    }

    #[test]
    fn test_concurrent_access() {
        let fst = Arc::new(parking_lot::RwLock::new(FST::new()));
        let threads = 10;
        let ops_per_thread = 1000;

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let fst = Arc::clone(&fst);

                thread::spawn(move || {
                    for i in 0..ops_per_thread {
                        let word = format!("word{}{}", t, i);
                        let doc_id = (t * ops_per_thread + i) as u64;
                        {
                            let mut fst = fst.write();
                            fst.insert(&word, create_posting(doc_id, 1));
                        }
                        {
                            let fst = fst.read();
                            assert!(fst.search(&word).is_some());
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_special_characters() {
        let mut fst = FST::new();

        let special_words = vec![
            "test\0test",
            "test\ntest",
            "test\rtest",
            "test!@#$%^&*()",
            "     ",
            "\t\t\t",
        ];

        for (i, word) in special_words.iter().enumerate() {
            fst.insert(word, create_posting(i as u64, 1));
            assert!(fst.search(word).is_some());
        }
    }

    #[test]
    fn test_rebuild_consistency() {
        let mut fst = FST::new();
        let word_count = 1000;

        // Insert words
        for i in 0..word_count {
            let word = format!("word{}", i);
            fst.insert(&word, create_posting(i as u64, 1));
        }

        // Delete some words
        for i in 0..word_count / 2 {
            let word = format!("word{}", i);
            fst.delete(&word);
        }

        // Force rebuild
        fst.rebuild();

        // Check consistency
        for i in 0..word_count {
            let word = format!("word{}", i);
            if i < word_count / 2 {
                assert!(fst.search(&word).is_none());
            } else {
                assert!(fst.search(&word).is_some());
            }
        }
    }

    #[test]
    fn test_posting_values() {
        let mut fst = FST::new();
        let posting = Posting {
            document_id: DocumentId(42),
            field_id: FieldId(7),
            positions: vec![1, 4, 9],
            term_frequency: 0.75,
            doc_length: 100,
        };

        fst.insert("test", posting);
        let retrieved = fst.search("test").unwrap();

        assert_eq!(retrieved.document_id.0, 42);
        assert_eq!(retrieved.field_id.0, 7);
        assert_eq!(retrieved.positions, vec![1, 4, 9]);
        assert_eq!(retrieved.term_frequency, 0.75);
        assert_eq!(retrieved.doc_length, 100);
    }

    #[test]
    fn test_multiple_fields() {
        let mut fst = FST::new();

        // Insert same word in different fields
        fst.insert("test", create_posting(1, 1));
        fst.insert("test", create_posting(1, 2));
        fst.insert("test", create_posting(1, 3));

        let result = fst.search("test").unwrap();
        assert!(result.field_id.0 > 0 && result.field_id.0 <= 3);
    }
}
