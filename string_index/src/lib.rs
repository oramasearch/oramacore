use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

use anyhow::Result;
use dictionary::{Dictionary, TermId};
use posting_storage::{PostingListId, PostingStorage};
// use radix_trie::Trie;
use ptrie::Trie;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use scorer::Scorer;
use serde::{Deserialize, Serialize};
use storage::Storage;
use types::{DocumentId, FieldId};

mod dictionary;
mod posting_storage;
pub mod scorer;

pub type DocumentBatch = HashMap<DocumentId, Vec<(FieldId, Vec<(String, Vec<String>)>)>>;

#[derive(Debug, Deserialize, Serialize)]
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
    term_frequency: usize,
}

pub struct StringIndex {
    tree: RwLock<Trie<u8, StringIndexValue>>,
    posting_storage: PostingStorage,
    dictionary: Dictionary,
    total_documents: AtomicUsize,
    total_document_length: AtomicUsize,
}

pub struct GlobalInfo {
    pub total_documents: usize,
    pub total_document_length: usize,
}

impl StringIndex {
    pub fn new(storage: Arc<Storage>) -> Self {
        StringIndex {
            tree: RwLock::new(Trie::new()),
            posting_storage: PostingStorage::new(storage),
            dictionary: Dictionary::new(),
            total_documents: AtomicUsize::new(0),
            total_document_length: AtomicUsize::new(0),
        }
    }

    pub fn get_total_documents(&self) -> usize {
        self.total_documents.load(Ordering::Relaxed)
    }

    pub fn search<S: Scorer + Sync>(
        &self,
        tokens: Vec<String>,
        search_on: Option<Vec<FieldId>>,
        boost: HashMap<FieldId, f32>,
        scorer: S,
    ) -> Result<HashMap<DocumentId, f32>> {
        let total_documents = match self.total_documents.load(Ordering::Relaxed) {
            0 => {
                println!("total_documents == 0");
                return Ok(Default::default());
            }
            total_documents => total_documents,
        };
        let total_document_length = match self.total_document_length.load(Ordering::Relaxed) {
            0 => {
                println!("total_document_length == 0");
                return Ok(Default::default());
            }
            total_document_length => total_document_length,
        };

        // let avg_doc_length = total_document_length / total_documents;

        let mut posting_list_ids_with_freq = Vec::<StringIndexValue>::new();
        let tree = self
            .tree
            .read()
            // TODO: better error handling
            .expect("Unable to read");
        for token in tokens {
            let a = tree.find_postfixes(token.bytes());
            posting_list_ids_with_freq.extend(a.into_iter().cloned());
        }

        let fields = search_on.as_ref();

        let global_info = GlobalInfo {
            total_documents,
            total_document_length,
        };

        posting_list_ids_with_freq
            .into_par_iter()
            .filter_map(|string_index_value| {
                let output = self
                    .posting_storage
                    .get(string_index_value.posting_list_id)
                    .ok();
                let posting = match output {
                    Some(v) => v,
                    None => return None,
                };

                let posting: Vec<_> = posting
                    .into_iter()
                    .filter(move |posting| {
                        fields
                            .map(|search_on| search_on.contains(&posting.field_id))
                            .unwrap_or(true)
                    })
                    .collect();

                Some((posting, string_index_value.term_frequency))
            })
            // Every thread perform on a separated hashmap
            .for_each(|(postings, total_token_count)| {
                let total_token_count = total_token_count as f32;
                for posting in postings {
                    let boost_per_field = *boost.get(&posting.field_id).unwrap_or(&1.0);
                    scorer.add_entry(&global_info, posting, total_token_count, boost_per_field);
                }
            });

        let scores = scorer.get_scores();

        Ok(scores)
    }

    pub fn insert_multiple(&self, data: DocumentBatch) -> Result<()> {
        self.total_documents
            .fetch_add(data.len(), Ordering::Relaxed);

        let dictionary = &self.dictionary;

        let t = data
            .into_par_iter()
            // Parallel
            .fold(
                HashMap::<TermId, Vec<Posting>>::new,
                |mut acc, (document_id, strings)| {
                    for (field_id, s) in strings {
                        let mut term_freqs: HashMap<String, HashMap<FieldId, Vec<usize>>> =
                            HashMap::new();

                        for (position, (original, stemmed)) in s.into_iter().enumerate() {
                            let entry = term_freqs.entry(original).or_default();
                            let field_entry = entry.entry(field_id).or_default();
                            field_entry.push(position);

                            for s in stemmed {
                                let entry = term_freqs.entry(s).or_default();
                                let field_entry = entry.entry(field_id).or_default();
                                field_entry.push(position);
                            }
                        }

                        let doc_length = term_freqs
                            .values()
                            .map(|field_freqs| {
                                field_freqs
                                    .values()
                                    .map(|positions| positions.len())
                                    .sum::<usize>()
                            })
                            .sum::<usize>();

                        for (term, field_positions) in term_freqs {
                            let term_id = dictionary.get_or_add(&term);
                            // println!("Term: {} -> {}", term, term_id.0);

                            let v = acc.entry(term_id).or_default();

                            let posting =
                                field_positions.into_iter().map(|(field_id, positions)| {
                                    let term_frequency = positions.len() as f32;

                                    Posting {
                                        document_id,
                                        field_id,
                                        positions,
                                        // original_term: term.clone(),
                                        term_frequency,
                                        doc_length: doc_length as u16,
                                    }
                                });
                            v.extend(posting);
                        }
                    }

                    acc
                },
            );

        let posting_per_term = t.reduce(
            HashMap::<TermId, Vec<Posting>>::new,
            // Merge the hashmap
            |mut acc, item| {
                for (term_id, postings) in item {
                    let vec = acc.entry(term_id).or_default();
                    vec.extend(postings.into_iter());
                }
                acc
            },
        );

        let mut postings_per_posting_list_id: HashMap<PostingListId, Vec<Vec<Posting>>> =
            HashMap::with_capacity(posting_per_term.len());
        let mut tree = self.tree.write().expect("Unable to write");
        // NB: We cannot parallelize the tree insertion yet :(
        // We could move the tree into a custom implementation to support parallelism
        // Once we resolve this issue, we StringIndex is thread safe!
        // TODO: move to custom implementation
        // For the time being, we can just use the sync tree
        for (term_id, postings) in posting_per_term {
            self.total_document_length.fetch_add(
                postings.iter().map(|p| p.positions.len()).sum::<usize>(),
                Ordering::Relaxed,
            );
            let number_of_occurence_of_term = postings.len();

            // Due to this implementation, we have a limitation
            // because we "forgot" the term. Here we have just the term_id
            // This invocation shouldn't exist at all:
            // we have the term on the top of this function
            // TODO: find a way to avoid this invocation
            let term = dictionary.retrive(term_id);

            let value = tree.get_mut(term.bytes());
            if let Some(value) = value {
                value.term_frequency += number_of_occurence_of_term;
                let vec = postings_per_posting_list_id
                    .entry(value.posting_list_id)
                    .or_default();
                vec.push(postings);
            } else {
                let posting_list_id = self.posting_storage.generate_new_id();
                tree.insert(
                    term.bytes(),
                    StringIndexValue {
                        posting_list_id,
                        term_frequency: number_of_occurence_of_term,
                    },
                );

                let vec = postings_per_posting_list_id
                    .entry(posting_list_id)
                    .or_default();
                vec.push(postings);
            }
        }

        postings_per_posting_list_id
            .into_par_iter()
            .map(|(k, v)| self.posting_storage.add_or_create(k, v))
            // TODO: handle error
            .all(|_| true);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use nlp::{locales::Locale, TextParser};

    use tempdir::TempDir;

    use crate::{scorer::bm25::BM25Score, DocumentId, FieldId, StringIndex};

    #[test]
    fn test_search() {
        let tmp_dir = TempDir::new("string_index_test").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();
        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = StringIndex::new(storage);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![
            (
                DocumentId(1),
                vec![(
                    FieldId(0),
                    "Yo, I'm from where Nicky Barnes got rich as fuck, welcome!".to_string(),
                )],
            ),
            (
                DocumentId(2),
                vec![(
                    FieldId(0),
                    "Welcome to Harlem, where you welcome to problems".to_string(),
                )],
            ),
            (
                DocumentId(3),
                vec![(
                    FieldId(0),
                    "Now bitches, they want to neuter me, niggas, they want to tutor me"
                        .to_string(),
                )],
            ),
        ]
        .into_iter()
        .map(|(doc_id, fields)| {
            let fields: Vec<_> = fields
                .into_iter()
                .map(|(field_id, data)| {
                    let tokens = parser.tokenize_and_stem(&data);
                    (field_id, tokens)
                })
                .collect();
            (doc_id, fields)
        })
        .collect();

        string_index.insert_multiple(batch).unwrap();

        let output = string_index
            .search(
                vec!["welcome".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert_eq!(output.len(), 2);

        let output = string_index
            .search(
                vec!["wel".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert_eq!(output.len(), 2);
    }
}
