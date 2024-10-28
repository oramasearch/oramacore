use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap}, sync::atomic::AtomicUsize,
};

use anyhow::Result;
use concurrent_radix_tree::{ConcurrentRadixTree, ConcurrentRadixTree2, InsertedOrUpdated};
use dictionary::{Dictionary, TermId};
use ordered_float::NotNan;
use posting_storage::{PostingListId, PostingStorage};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use string_ultils::{tokenize, tokenize_and_stem, Language, Parser};

mod dictionary;
mod posting_storage;
mod concurrent_radix_tree;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct DocumentId(pub usize);
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldId(pub usize);

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Posting {
    pub document_id: DocumentId,
    pub field_id: FieldId,
    pub positions: Vec<usize>,
    pub term_frequency: f32,
    pub doc_length: u16,
}

#[derive(Debug, Clone)]
pub struct StringIndexValue {
    posting_list_id: PostingListId,
    term_frequency: usize
}

#[derive(Debug)]
pub struct StringIndex {
    index: ConcurrentRadixTree2<StringIndexValue>,
    posting_storage: PostingStorage,
    parser: Parser,
    dictionary: Dictionary,
    total_documents: AtomicUsize,
    total_document_length: AtomicUsize,
}

impl StringIndex {
    pub fn new(base_path: String) -> Self {
        StringIndex {
            index: ConcurrentRadixTree2::new(),
            posting_storage: PostingStorage::new(format!("{}/posting_storage", base_path)).unwrap(),
            parser: Parser::from_language(Language::English),
            dictionary: Dictionary::new(),
            total_documents: AtomicUsize::new(0),
            total_document_length: AtomicUsize::new(0),
        }
    }

    pub fn search(&self, term: &str, limit: usize, boost: f32) -> Result<Vec<(DocumentId, f32)>> {
        let total_documents = match self.total_documents.load(std::sync::atomic::Ordering::Relaxed) {
            0 => {
                return {
                    println!("total_documents == 0");
                    Ok(vec![])
                }
            }
            total_documents => total_documents as f32,
        };
        let total_document_length = match self.total_document_length.load(std::sync::atomic::Ordering::Relaxed) {
            0 => {
                println!("total_document_length == 0");
                return Ok(vec![]);
            }
            total_document_length => total_document_length as f32,
        };

        let avg_doc_length = total_document_length / total_documents;

        let tokens = tokenize(term, &self.parser.tokenizer);

        let mut posting_list_ids_with_freq = Vec::<StringIndexValue>::new();
        for token in tokens {
            let string_index_value = self.index.get(&token);

            if let Some(string_index_value) = string_index_value {
                posting_list_ids_with_freq.push(string_index_value);
            } else {
                eprintln!("Token not found inside index: {}", token);
            }
        }

        // println!("posting_list_ids_with_freq: {:#?}", posting_list_ids_with_freq);

        let scores = posting_list_ids_with_freq
            .into_par_iter()
            .filter_map(|string_index_value| {
                self.posting_storage.get(
                    string_index_value.posting_list_id,
                    // BAD: term_frequency is not used inside posting_storage
                    // But we need after, so here forward it.
                    // TODO: We need to find a way to avoid this.
                    string_index_value.term_frequency,
                ).ok()
            })
            // Every thread perform on a separated hashmap
            .fold(HashMap::<DocumentId, f32>::new, |mut acc, (postings, freq)| {
                let freq = freq as f32;
                for posting in postings {
                    let term_frequency = posting.term_frequency;
                    let doc_length = posting.doc_length as f32;

                    let idf = ((total_documents - freq + 0.5_f32)
                        / (freq + 0.5_f32))
                        .ln_1p();
                    let score =
                        calculate_score(term_frequency, idf, doc_length, avg_doc_length, boost);

                    let doc_score = acc.entry(posting.document_id)
                        .or_default();
                    *doc_score += score;
                }
                acc
            })
            // And later we merge all the hashmaps
            .reduce(HashMap::<DocumentId, f32>::new, |mut acc, item| {
                for (document_id, score) in item {
                    let doc_score = acc.entry(document_id).or_default();
                    *doc_score += score;
                }
                acc
            });

        let docs = top_n(scores, limit);

        Ok(docs)
    }

    fn insert_single(&self, document_id: DocumentId, fields: Vec<(FieldId, String)>, postings_per_posting_list_id: &mut HashMap<PostingListId, Vec<Vec<Posting>>>) {
        self.total_documents.fetch_add(1, std::sync::atomic::Ordering::AcqRel);

        let mut doc_length: usize = 0;
        let mut term_freqs: HashMap<String, HashMap<FieldId, Vec<usize>>> = HashMap::new();
        for (field_id, value) in fields {
            let tokens = tokenize_and_stem(&value, &self.parser);

            for (position, (original, stemmed)) in tokens.enumerate() {
                self.dictionary.get_or_add(&original);
                self.dictionary.get_or_add(&stemmed);

                let is_equal = original == stemmed;

                let entry = term_freqs.entry(original).or_default();
                let field_entry = entry.entry(field_id).or_default();
                field_entry.push(position);
                doc_length += 1;

                if !is_equal {
                    let entry = term_freqs.entry(stemmed).or_default();
                    let field_entry = entry.entry(field_id).or_default();
                    field_entry.push(position);
                    doc_length += 1;
                }
            }
        }

        // let  = HashMap::with_capacity(term_freqs.len());
        for (term, field_positions) in term_freqs {
            // let term_id = self.dictionary.get_or_add(&term);

            let postings: Vec<_> = field_positions.into_iter().map(|(field_id, positions)| {
                let term_frequency = positions.len() as f32;

                // println!("term: {}, document_id: {:?}", term, document_id);
                Posting {
                    document_id,
                    field_id,
                    positions,
                    term_frequency,
                    doc_length: doc_length as u16,
                }
            }).collect();

            self.total_document_length.fetch_add(
                postings.iter().map(|p| p.positions.len()).sum::<usize>(),
                std::sync::atomic::Ordering::AcqRel,
            );

            let number_of_occurence_of_term = postings.len();

            self.index.insert_or_update(term, 
                |value| {
                    match value {
                        None => {
                            let posting_list_id = self.posting_storage.generate_new_id();
                            let vec = postings_per_posting_list_id.entry(posting_list_id)
                            .or_default();
                            vec.push(postings);

                            InsertedOrUpdated::Inserted(StringIndexValue {
                                posting_list_id,
                                term_frequency: number_of_occurence_of_term
                            })
                        },
                        Some(value) => {
                            value.term_frequency += number_of_occurence_of_term;
                            let vec = postings_per_posting_list_id.entry(value.posting_list_id)
                            .or_default();
                            vec.push(postings);

                            InsertedOrUpdated::Updated
                        }
                    }
            });
        }
    }

    fn foo(&self) -> Vec<Posting> {
        let output = self.index.get("welcome").unwrap();
        let posting_list_id = output.posting_list_id;
        let postings = self.posting_storage.get(posting_list_id, 5).unwrap().0;
        postings
    }

    pub fn insert_multiple(
        &mut self,
        data: Vec<(DocumentId, Vec<(FieldId, String)>)>,
    ) -> Result<()> {
        let postings_per_posting_list_id = data.into_par_iter()
            .fold(HashMap::<PostingListId, Vec<Vec<Posting>>>::new, |mut acc, (document_id, fields)| {
                self.insert_single(document_id, fields, &mut acc);
                acc
            })
            .reduce(HashMap::<PostingListId, Vec<Vec<Posting>>>::new, |mut acc, item| {
                for (k, v) in item {
                    let vec = acc.entry(k).or_default();
                    vec.extend(v);
                }
                acc
            });

        postings_per_posting_list_id
            .into_par_iter()
            .for_each(|(k, v)| {
                self.posting_storage.add_or_create(
                    k, v
                ).unwrap();
            });
            
        Ok(())
    }
}

fn calculate_score(tf: f32, idf: f32, doc_length: f32, avg_doc_length: f32, boost: f32) -> f32 {
    let k1 = 1.5;
    let b = 0.75;
    let numerator = tf * (k1 + 1.0);
    let denominator = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_length));
    idf * (numerator / denominator) * boost
}

fn top_n(map: HashMap<DocumentId, f32>, n: usize) -> Vec<(DocumentId, f32)> {
    // A min-heap of size `n` to keep track of the top N elements
    let mut heap: BinaryHeap<Reverse<(NotNan<f32>, DocumentId)>> = BinaryHeap::with_capacity(n);

    for (key, value) in map {
        // Insert into the heap if it's not full, or replace the smallest element if the current one is larger
        if heap.len() < n {
            heap.push(Reverse((NotNan::new(value).unwrap(), key)));
        } else if let Some(Reverse((min_value, _))) = heap.peek() {
            if value > *min_value.as_ref() {
                heap.pop();
                heap.push(Reverse((NotNan::new(value).unwrap(), key)));
            }
        }
    }

    // Collect results into a sorted Vec (optional sorting based on descending values)
    let result: Vec<(DocumentId, f32)> = heap
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse((value, key))| (key, value.into_inner()))
        .collect();

    // TODO: check is this `reverse` is needed
    // result.reverse();
    result
}

#[cfg(test)]
mod tests {
    use crate::{DocumentId, FieldId, StringIndex};

    #[test]
    fn test_foo() {
        let batch = vec![
            (
                DocumentId(0),
                vec![(
                    FieldId(0),
                    "Yo, I'm from where Nicky Barnes got rich as fuck, welcome!".to_string(),
                )],
            ),
            (
                DocumentId(1),
                vec![(
                    FieldId(0),
                    "Welcome to Harlem, where you welcome to problems".to_string(),
                )],
            ),
            (
                DocumentId(2),
                vec![(
                    FieldId(0),
                    "Now bitches, they want to neuter me, niggas, they want to tutor me".to_string(),
                )],
            ),
        ];

        let tmpfile = tempfile::tempdir().unwrap();
        let mut string_index = StringIndex::new(tmpfile.path().to_string_lossy().to_string());
        string_index.insert_multiple(batch).unwrap();

        let output = string_index.foo();
        println!("{:?}", output);

        assert_eq!(output.len(), 2);

        // let output = string_index.search("welcome", 10, 1.0).unwrap();
        // println!("{string_index:#?}");
        // println!("{:?}", "welcome".encode());
        // println!("{output:?}");
        // assert_eq!(output.len(), 2);
    }
}
