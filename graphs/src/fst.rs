use serde::{Deserialize, Serialize};
use string_index::Posting;
use types::{DocumentId, FieldId};

#[derive(Serialize, Deserialize)]
struct Transition {
    input: char,
    output: Option<Posting>,
    next_state: usize,
}

#[derive(Serialize, Deserialize)]
struct State {
    transitions: Vec<Transition>,
}

#[derive(Serialize, Deserialize)]
pub struct FST {
    states: Vec<State>,
}

impl FST {
    pub fn new() -> Self {
        Self {
            states: vec![State {
                transitions: Vec::new(),
            }],
        }
    }

    pub fn insert(&mut self, word: &str, posting: Posting) {
        let mut current_state = 0;
        let chars: Vec<char> = word.chars().collect();

        for (i, &c) in chars.iter().enumerate() {
            let mut next_state = None;

            if let Some(transition) = self.states[current_state]
                .transitions
                .iter()
                .find(|t| t.input == c)
            {
                next_state = Some(transition.next_state);
            }

            let next_state = match next_state {
                Some(state) => state,
                None => {
                    let new_state = self.add_state();
                    self.states[current_state].transitions.push(Transition {
                        input: c,
                        output: None,
                        next_state: new_state,
                    });
                    new_state
                }
            };

            if i == chars.len() - 1 {
                self.states[current_state]
                    .transitions
                    .retain(|t| t.input != c);
                self.states[current_state].transitions.push(Transition {
                    input: c,
                    output: Some(posting.clone()),
                    next_state,
                });
            }

            current_state = next_state;
        }

        if word.is_empty() {
            self.states[0].transitions.push(Transition {
                input: '\0',
                output: Some(posting),
                next_state: 0,
            });
        }
    }

    pub fn search(&self, input: &str) -> Option<Posting> {
        let mut current_state = 0;
        let chars: Vec<char> = input.chars().collect();

        if input.is_empty() {
            return self.states[0]
                .transitions
                .iter()
                .find(|t| t.input == '\0')
                .and_then(|t| t.output.clone());
        }

        for (i, &c) in chars.iter().enumerate() {
            if let Some(transition) = self.states[current_state]
                .transitions
                .iter()
                .find(|t| t.input == c)
            {
                if i == chars.len() - 1 {
                    return transition.output.clone();
                }
                current_state = transition.next_state;
            } else {
                return None;
            }
        }
        None
    }

    fn add_state(&mut self) -> usize {
        self.states.push(State {
            transitions: Vec::new(),
        });
        self.states.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_posting(
        document_id: u32,
        field_id: u32,
        positions: Vec<usize>,
        term_frequency: f32,
        doc_length: u16,
    ) -> Posting {
        Posting {
            document_id: DocumentId(document_id as u64),
            field_id: FieldId(field_id as u16),
            positions,
            term_frequency,
            doc_length,
        }
    }

    #[test]
    fn test_insert_and_search_posting() {
        let mut fst = FST::new();
        let pst = create_test_posting(1, 101, vec![3, 10, 42], 2.5, 120);

        fst.insert("example", pst.clone());
        let result = fst.search("example");
        assert!(result.is_some());
        let found_posting = result.unwrap();
        assert_eq!(found_posting.document_id, pst.document_id);
        assert_eq!(found_posting.field_id, pst.field_id);
        assert_eq!(found_posting.positions, pst.positions);
        assert_eq!(found_posting.term_frequency, pst.term_frequency);
        assert_eq!(found_posting.doc_length, pst.doc_length);
    }

    #[test]
    fn test_search_nonexistent_word() {
        let mut fst = FST::new();
        let posting = create_test_posting(2, 102, vec![5, 15, 25], 1.8, 100);

        fst.insert("word", posting);
        let result = fst.search("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_insert_with_overlapping_prefixes() {
        let mut fst = FST::new();
        let posting1 = create_test_posting(3, 103, vec![1, 2, 3], 3.0, 80);
        let posting2 = create_test_posting(4, 104, vec![4, 5, 6], 2.0, 90);

        fst.insert("test", posting1.clone());
        fst.insert("testing", posting2.clone());

        let result1 = fst.search("test");
        let result2 = fst.search("testing");

        assert!(result1.is_some());
        assert!(result2.is_some());

        let found_posting1 = result1.unwrap();
        let found_posting2 = result2.unwrap();

        assert_eq!(found_posting1.document_id, posting1.document_id);
        assert_eq!(found_posting2.document_id, posting2.document_id);
    }
}
