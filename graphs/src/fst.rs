use dashmap::DashMap;
use std::{mem::size_of, sync::Arc};
use string_index::Posting;

const MAX_TRANSITIONS: usize = 32;
const CACHE_LINE: usize = 64;

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
}

impl FST {
    pub fn new() -> Self {
        let mut transitions = Vec::with_capacity(1_000_000);
        transitions.push(TransitionBlock::new());

        Self {
            transitions,
            postings: Box::new(DashMap::with_capacity_and_shard_amount(1_000_000, 32)),
            free_blocks: Vec::with_capacity(10_000),
        }
    }

    #[inline]
    pub fn insert(&mut self, word: &str, posting: Posting) -> u32 {
        let bytes = word.as_bytes();
        if bytes.is_empty() {
            return 0;
        }

        let posting_id = self.store_posting(posting);
        let mut current_block = 0;

        if bytes.len() == 1 {
            let block = &mut self.transitions[current_block];
            let pos = if let Some(pos) = block.find_char(bytes[0]) {
                pos
            } else {
                block.add_transition(bytes[0], 0)
            };
            block.posting_ids[pos] = posting_id;
            return posting_id;
        }

        let last_idx = bytes.len() - 1;
        for (i, &b) in bytes[..last_idx].iter().enumerate() {
            let block = &self.transitions[current_block];
            let (next_block, create_new) = if let Some(pos) = block.find_char(b) {
                (block.next_states[pos] as usize, false)
            } else {
                (self.allocate_block(), true)
            };

            if create_new {
                let block = &mut self.transitions[current_block];
                block.add_transition(b, next_block as u32);
            }

            current_block = next_block;
        }

        let b = bytes[last_idx];
        let block = &mut self.transitions[current_block];
        let pos = if let Some(pos) = block.find_char(b) {
            pos
        } else {
            block.add_transition(b, 0)
        };
        block.posting_ids[pos] = posting_id;

        posting_id
    }

    #[inline]
    pub fn search(&self, word: &str) -> Option<Arc<Posting>> {
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
    pub fn batch_insert(&mut self, entries: Vec<(&str, Posting)>) {
        self.transitions
            .reserve(entries.iter().map(|(w, _)| w.len()).sum());

        let mut entries = entries;
        entries.sort_unstable_by_key(|(w, _)| w.len());

        for (word, posting) in entries {
            self.insert(word, posting);
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
