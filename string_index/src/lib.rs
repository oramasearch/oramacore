use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

use anyhow::Result;
use dictionary::{Dictionary, TermId};
use fst::{Automaton, IntoStreamer, Map, MapBuilder, Streamer};
use posting_storage::{PostingListId, PostingStorage};
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

#[derive(Debug, Clone, Copy)]
pub struct StringIndexValue {
    pub posting_list_id: PostingListId,
    pub term_frequency: usize,
}

pub struct StringIndex {
    fst_map: RwLock<Option<Map<Vec<u8>>>>,
    temp_map: RwLock<HashMap<String, StringIndexValue>>,
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
            fst_map: RwLock::new(None),
            temp_map: RwLock::new(HashMap::new()),
            posting_storage: PostingStorage::new(storage),
            dictionary: Dictionary::new(),
            total_documents: AtomicUsize::new(0),
            total_document_length: AtomicUsize::new(0),
        }
    }

    fn build_fst(&self) -> Result<Map<Vec<u8>>> {
        let entries: Vec<_> = {
            let temp_map = self.temp_map.read().unwrap();
            temp_map
                .iter()
                .map(|(key, value)| {
                    (
                        key.clone(),
                        ((value.posting_list_id.0 as u64) << 32) | (value.term_frequency as u64),
                    )
                })
                .collect()
        };

        let mut sorted_entries = entries;
        sorted_entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        let mut builder = MapBuilder::memory();

        for (key, value) in sorted_entries {
            builder.insert(key.as_bytes(), value)?;
        }

        let fst = builder.into_map();

        Ok(fst)
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
                return Ok(Default::default());
            }
            total_documents => total_documents,
        };

        let total_document_length = match self.total_document_length.load(Ordering::Relaxed) {
            0 => {
                return Ok(Default::default());
            }
            total_document_length => total_document_length,
        };

        let mut all_postings = Vec::new();

        if let Some(fst) = self.fst_map.read().unwrap().as_ref() {
            for token in &tokens {
                let automaton = fst::automaton::Str::new(token).starts_with();
                let mut stream = fst.search(automaton).into_stream();

                while let Some(result) = stream.next() {
                    let (_, packed_value) = result;
                    let posting_list_id = PostingListId(((packed_value >> 32) as u32) as usize);
                    let term_frequency = (packed_value & 0xFFFFFFFF) as usize;

                    match self.posting_storage.get(posting_list_id) {
                        Ok(postings) => {
                            all_postings.push((postings, term_frequency));
                        }
                        Err(e) => {}
                    }
                }
            }
        }

        let fields = search_on.as_ref();
        let global_info = GlobalInfo {
            total_documents,
            total_document_length,
        };

        let mut filtered_postings = Vec::new();
        for (postings, term_freq) in all_postings {
            for posting in postings {
                if fields
                    .map(|search_on| search_on.contains(&posting.field_id))
                    .unwrap_or(true)
                {
                    filtered_postings.push((posting, term_freq));
                }
            }
        }

        for (posting, term_freq) in filtered_postings {
            let boost_per_field = *boost.get(&posting.field_id).unwrap_or(&1.0);
            scorer.add_entry(&global_info, posting, term_freq as f32, boost_per_field);
        }

        let scores = scorer.get_scores();

        Ok(scores)
    }

    pub fn insert_multiple(&self, data: DocumentBatch) -> Result<()> {
        self.total_documents
            .fetch_add(data.len(), Ordering::Relaxed);

        let dictionary = &self.dictionary;

        let posting_per_term = data
            .into_par_iter()
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
                            let v = acc.entry(term_id).or_default();

                            let posting =
                                field_positions.into_iter().map(|(field_id, positions)| {
                                    let term_frequency = positions.len() as f32;

                                    Posting {
                                        document_id,
                                        field_id,
                                        positions,
                                        term_frequency,
                                        doc_length: doc_length as u16,
                                    }
                                });
                            v.extend(posting);
                        }
                    }

                    acc
                },
            )
            .reduce(HashMap::<TermId, Vec<Posting>>::new, |mut acc, item| {
                for (term_id, postings) in item {
                    let vec = acc.entry(term_id).or_default();
                    vec.extend(postings.into_iter());
                }
                acc
            });

        let mut postings_per_posting_list_id: HashMap<PostingListId, Vec<Vec<Posting>>> =
            HashMap::with_capacity(posting_per_term.len());
        let mut temp_map = self.temp_map.write().unwrap();

        for (term_id, postings) in posting_per_term {
            self.total_document_length.fetch_add(
                postings.iter().map(|p| p.positions.len()).sum::<usize>(),
                Ordering::Relaxed,
            );
            let number_of_occurence_of_term = postings.len();

            let term = dictionary.retrive(term_id);

            if let Some(value) = temp_map.get_mut(&term) {
                value.term_frequency += number_of_occurence_of_term;
                let vec = postings_per_posting_list_id
                    .entry(value.posting_list_id)
                    .or_default();
                vec.push(postings);
            } else {
                let posting_list_id = self.posting_storage.generate_new_id();
                temp_map.insert(
                    term,
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

        drop(temp_map);
        let new_fst = self.build_fst()?;

        {
            let mut fst_map = self.fst_map.write().unwrap();
            *fst_map = Some(new_fst);
        }

        for (k, v) in postings_per_posting_list_id {
            self.posting_storage.add_or_create(k, v)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{scorer::bm25::BM25Score, DocumentId, FieldId, StringIndex};
    use nlp::{locales::Locale, TextParser};
    use std::{collections::HashMap, sync::Arc};
    use tempdir::TempDir;

    #[test]
    fn test_empty_search_query() {
        let tmp_dir = TempDir::new("string_index_test_empty_search").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();

        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = StringIndex::new(storage);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(FieldId(0), "This is a test document.".to_string())],
        )]
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
                vec![], // Empty search tokens
                None,
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty for empty query"
        );
    }

    #[test]
    fn test_search_nonexistent_term() {
        let tmp_dir = TempDir::new("string_index_test_nonexistent_term").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();

        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = StringIndex::new(storage);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(FieldId(0), "This is a test document.".to_string())],
        )]
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
                vec!["nonexistent".to_string()], // Term does not exist
                None,
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty for non-existent term"
        );
    }

    #[test]
    fn test_insert_empty_document() {
        let tmp_dir = TempDir::new("string_index_test_empty_document").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();

        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = StringIndex::new(storage);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(FieldId(0), "".to_string())], // Empty document content
        )]
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

        // Search for any term, should get empty result
        let output = string_index
            .search(
                vec!["test".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty when only empty documents are indexed"
        );
    }

    #[test]
    fn test_search_with_field_filter() {
        let tmp_dir = TempDir::new("string_index_test_field_filter").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();

        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = StringIndex::new(storage);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![
                (FieldId(0), "This is a test in field zero.".to_string()),
                (FieldId(1), "Another test in field one.".to_string()),
            ],
        )]
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
                vec!["test".to_string()],
                Some(vec![FieldId(0)]), // Search only in FieldId(0)
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document when searching in FieldId(0)"
        );

        let output = string_index
            .search(
                vec!["test".to_string()],
                Some(vec![FieldId(1)]),
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document when searching in FieldId(1)"
        );

        let output = string_index
            .search(
                vec!["test".to_string()],
                Some(vec![FieldId(2)]),
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert!(
            output.is_empty(),
            "Should not find any documents when searching in non-existent FieldId"
        );
    }

    #[test]
    fn test_search_with_boosts() {
        let tmp_dir = TempDir::new("string_index_test_boosts").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();

        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = StringIndex::new(storage);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![
            (
                DocumentId(1),
                vec![(FieldId(0), "Important content in field zero.".to_string())],
            ),
            (
                DocumentId(2),
                vec![(
                    FieldId(1),
                    "Less important content in field one.".to_string(),
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

        let mut boost = HashMap::new();
        boost.insert(FieldId(0), 2.0);

        let output = string_index
            .search(
                vec!["content".to_string()],
                None,
                boost,
                BM25Score::default(),
            )
            .unwrap();

        assert_eq!(output.len(), 2, "Should find both documents");

        let score_doc1 = output.get(&DocumentId(1)).unwrap();
        let score_doc2 = output.get(&DocumentId(2)).unwrap();

        assert!(
            score_doc1 > score_doc2,
            "Document with boosted field should have higher score"
        );
    }

    #[test]
    fn test_insert_document_with_stop_words_only() {
        let tmp_dir = TempDir::new("string_index_test_stop_words").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();

        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = StringIndex::new(storage);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(FieldId(0), "the and but or".to_string())], // Only stop words
        )]
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

        // Search for any term, should get empty result since only stop words are indexed
        let output = string_index
            .search(
                vec!["the".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty when only stop words are indexed"
        );
    }

    #[test]
    fn test_search_on_empty_index() {
        let tmp_dir = TempDir::new("string_index_test_empty_index").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();

        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = StringIndex::new(storage);

        let output = string_index
            .search(
                vec!["test".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty when index is empty"
        );
    }

    #[test]
    fn test_concurrent_insertions() {
        use std::thread;

        let tmp_dir = TempDir::new("string_index_test_concurrent_inserts").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();

        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = Arc::new(StringIndex::new(storage));

        let string_index_clone1 = Arc::clone(&string_index);
        let string_index_clone2 = Arc::clone(&string_index);

        let handle1 = thread::spawn(move || {
            let parser = TextParser::from_language(Locale::EN);
            let batch: HashMap<_, _> = vec![(
                DocumentId(1),
                vec![(
                    FieldId(0),
                    "Concurrent insertion test document one.".to_string(),
                )],
            )]
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

            string_index_clone1.insert_multiple(batch).unwrap();
        });

        let handle2 = thread::spawn(move || {
            let parser = TextParser::from_language(Locale::EN);
            let batch: HashMap<_, _> = vec![(
                DocumentId(2),
                vec![(
                    FieldId(0),
                    "Concurrent insertion test document two.".to_string(),
                )],
            )]
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

            string_index_clone2.insert_multiple(batch).unwrap();
        });

        handle1.join().unwrap();
        handle2.join().unwrap();

        // After concurrent insertions, search for "concurrent"
        let parser = TextParser::from_language(Locale::EN);
        let search_tokens = parser
            .tokenize_and_stem("concurrent")
            .into_iter()
            .map(|(original, _)| original)
            .collect::<Vec<_>>();

        let output = string_index
            .search(
                search_tokens,
                None,
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert_eq!(
            output.len(),
            2,
            "Should find both documents after concurrent insertions"
        );
    }

    #[test]
    fn test_large_documents() {
        let tmp_dir = TempDir::new("string_index_test_large_documents").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();

        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = StringIndex::new(storage);
        let parser = TextParser::from_language(Locale::EN);

        let large_text = "word ".repeat(10000); // Create a large document

        let batch: HashMap<_, _> = vec![(DocumentId(1), vec![(FieldId(0), large_text.clone())])]
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

        // Search for "word"
        let output = string_index
            .search(
                vec!["word".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document containing the large text"
        );
    }

    #[test]
    fn test_high_term_frequency() {
        let tmp_dir = TempDir::new("string_index_test_high_term_frequency").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();

        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = StringIndex::new(storage);
        let parser = TextParser::from_language(Locale::EN);

        let repeated_word = "repeat ".repeat(1000); // High term frequency

        let batch: HashMap<_, _> = vec![(DocumentId(1), vec![(FieldId(0), repeated_word.clone())])]
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

        // Search for "repeat"
        let output = string_index
            .search(
                vec!["repeat".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
            )
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document with high term frequency"
        );
    }

    #[test]
    fn test_term_positions() {
        let tmp_dir = TempDir::new("string_index_test_term_positions").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();
        let storage = Arc::new(crate::Storage::from_path(&tmp_dir));
        let string_index = StringIndex::new(storage);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(
                FieldId(0),
                "quick brown fox jumps over the lazy dog".to_string(),
            )],
        )]
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
    }
}
