use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

use anyhow::Result;
use dictionary::{Dictionary, TermId};
use fst::{Automaton, IntoStreamer, Map, MapBuilder, Streamer};
use posting_storage::{PostingListId, PostingStorage};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use scorer::Scorer;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};

use crate::{collection_manager::FieldId, document_storage::DocumentId};

mod dictionary;
mod posting_storage;
pub mod scorer;

pub type DocumentBatch = HashMap<DocumentId, Vec<(FieldId, Vec<(String, Vec<String>)>)>>;

#[derive(Debug, Deserialize, Serialize, Clone)]
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
    insert_mutex: Mutex<()>,
}

pub struct GlobalInfo {
    pub total_documents: usize,
    pub total_document_length: usize,
}

impl StringIndex {
    pub fn new(id_generator: Arc<AtomicU64>) -> Self {
        StringIndex {
            fst_map: RwLock::new(None),
            temp_map: RwLock::new(HashMap::new()),
            posting_storage: PostingStorage::new(id_generator),
            dictionary: Dictionary::new(),
            total_documents: AtomicUsize::new(0),
            total_document_length: AtomicUsize::new(0),
            insert_mutex: Mutex::new(()),
        }
    }

    async fn build_fst(&self) -> Result<Map<Vec<u8>>> {
        let entries: Vec<_> = {
            let temp_map = self.temp_map.read().await;
            temp_map
                .iter()
                .map(|(key, value)| {
                    (
                        key.clone(),
                        (value.posting_list_id.0 << 32) | (value.term_frequency as u64),
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

    pub async fn search<S: Scorer + Sync>(
        &self,
        tokens: Vec<String>,
        search_on: Option<Vec<FieldId>>,
        boost: HashMap<FieldId, f32>,
        scorer: S,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
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

        if let Some(fst) = self.fst_map.read().await.as_ref() {
            for token in &tokens {
                let automaton = fst::automaton::Str::new(token).starts_with();
                let mut stream = fst.search(automaton).into_stream();

                while let Some(result) = stream.next() {
                    let (_, packed_value) = result;
                    let posting_list_id = PostingListId(((packed_value >> 32) as u32) as u64);
                    let term_frequency = (packed_value & 0xFFFFFFFF) as usize;

                    if let Ok(postings) = self.posting_storage.get(posting_list_id) {
                        all_postings.push((postings, term_frequency));
                    }
                }
            }
        }

        let fields = search_on.as_ref();
        let global_info = GlobalInfo {
            total_documents,
            total_document_length,
        };

        let in_filtered = move |posting: &Posting| -> bool {
            if let Some(filtered_doc_ids) = filtered_doc_ids {
                filtered_doc_ids.contains(&posting.document_id)
            } else {
                true
            }
        };

        let mut filtered_postings: HashMap<DocumentId, Vec<(Posting, usize)>> = HashMap::new();
        for (postings, term_freq) in all_postings {
            for posting in postings.into_iter().filter(in_filtered) {
                if fields
                    .map(|search_on| search_on.contains(&posting.field_id))
                    .unwrap_or(true)
                {
                    filtered_postings
                        .entry(posting.document_id)
                        .or_default()
                        .push((posting, term_freq));
                }
            }
        }

        let mut exact_match_documents = Vec::new();

        for (document_id, postings) in &filtered_postings {
            let mut token_positions: Vec<Vec<usize>> = postings
                .iter()
                .map(|(posting, _)| posting.positions.clone())
                .collect();

            token_positions
                .iter_mut()
                .for_each(|positions| positions.sort_unstable());

            if self.is_phrase_match(&token_positions) {
                exact_match_documents.push(*document_id);
            }

            for (posting, term_freq) in postings {
                let boost_per_field = *boost.get(&posting.field_id).unwrap_or(&1.0);
                scorer.add_entry(
                    &global_info,
                    posting.clone(),
                    *term_freq as f32,
                    boost_per_field,
                );
            }
        }

        let mut scores = scorer.get_scores();

        let exact_match_boost = 25.0; // @todo: make this configurable.
        for document_id in exact_match_documents {
            if let Some(score) = scores.get_mut(&document_id) {
                *score *= exact_match_boost;
            }
        }

        Ok(scores)
    }

    pub async fn insert_multiple(&self, data: DocumentBatch) -> Result<()> {
        let _lock = self.insert_mutex.lock().await;

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
        let mut temp_map = self.temp_map.write().await;

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
        let new_fst = self.build_fst().await?;

        {
            let mut fst_map = self.fst_map.write().await;
            *fst_map = Some(new_fst);
        }

        for (k, v) in postings_per_posting_list_id {
            self.posting_storage.add_or_create(k, v)?;
        }

        Ok(())
    }

    fn is_phrase_match(&self, token_positions: &[Vec<usize>]) -> bool {
        if token_positions.is_empty() {
            return false;
        }

        let position_sets: Vec<std::collections::HashSet<usize>> = token_positions
            .iter()
            .skip(1)
            .map(|positions| positions.iter().copied().collect())
            .collect();

        for &start_pos in &token_positions[0] {
            let mut current_pos = start_pos;
            let mut matched = true;

            for positions in &position_sets {
                let next_pos = current_pos + 1;
                if positions.contains(&next_pos) {
                    current_pos = next_pos;
                } else {
                    matched = false;
                    break;
                }
            }

            if matched {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use futures::{future::join_all, FutureExt};

    use crate::{
        collection_manager::FieldId,
        document_storage::DocumentId,
        indexes::string::{scorer::bm25::BM25Score, StringIndex},
        nlp::{locales::Locale, TextParser},
    };
    use std::{collections::HashMap, sync::Arc};

    #[tokio::test]
    async fn test_empty_search_query() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
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

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(vec![], None, Default::default(), BM25Score::default(), None)
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty for empty query"
        );
    }

    #[tokio::test]
    async fn test_search_nonexistent_term() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
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

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["nonexistent".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty for non-existent term"
        );
    }

    #[tokio::test]
    async fn test_insert_empty_document() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(DocumentId(1), vec![(FieldId(0), "".to_string())])]
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

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["test".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty when only empty documents are indexed"
        );
    }

    #[tokio::test]
    async fn test_search_with_field_filter() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
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

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["test".to_string()],
                Some(vec![FieldId(0)]),
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
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
                None,
            )
            .await
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
                None,
            )
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Should not find any documents when searching in non-existent FieldId"
        );
    }

    #[tokio::test]
    async fn test_search_with_boosts() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
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

        string_index.insert_multiple(batch).await.unwrap();

        let mut boost = HashMap::new();
        boost.insert(FieldId(0), 2.0);

        let output = string_index
            .search(
                vec!["content".to_string()],
                None,
                boost,
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(output.len(), 2, "Should find both documents");

        let score_doc1 = output.get(&DocumentId(1)).unwrap();
        let score_doc2 = output.get(&DocumentId(2)).unwrap();

        assert!(
            score_doc1 > score_doc2,
            "Document with boosted field should have higher score"
        );
    }

    #[tokio::test]
    async fn test_insert_document_with_stop_words_only() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(FieldId(0), "the and but or".to_string())],
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

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["the".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty when only stop words are indexed"
        );
    }

    #[tokio::test]
    async fn test_search_on_empty_index() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);

        let output = string_index
            .search(
                vec!["test".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty when index is empty"
        );
    }

    #[tokio::test]
    async fn test_concurrent_insertions() {
        let id_geneator = Arc::new(Default::default());
        let string_index = Arc::new(StringIndex::new(id_geneator));

        let string_index_clone1 = Arc::clone(&string_index);
        let string_index_clone2 = Arc::clone(&string_index);

        let handle1 = async {
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

            string_index_clone1.insert_multiple(batch).await.unwrap();
        }
        .boxed();

        let handle2 = async {
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

            string_index_clone2.insert_multiple(batch).await.unwrap();
        }
        .boxed();

        join_all(vec![handle1, handle2]).await;

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
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            output.len(),
            2,
            "Should find both documents after concurrent insertions"
        );
    }

    #[tokio::test]
    async fn test_large_documents() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let large_text = "word ".repeat(10000);

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

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["word".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document containing the large text"
        );
    }

    #[tokio::test]
    async fn test_high_term_frequency() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let repeated_word = "repeat ".repeat(1000);

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

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["repeat".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document with high term frequency"
        );
    }

    #[tokio::test]
    async fn test_term_positions() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
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

        string_index.insert_multiple(batch).await.unwrap();
    }

    #[tokio::test]
    async fn test_exact_phrase_match() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(FieldId(0), "5200 mAh battery in disguise".to_string())],
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

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec![
                    "5200".to_string(),
                    "mAh".to_string(),
                    "battery".to_string(),
                    "in".to_string(),
                    "disguise".to_string(),
                ],
                Some(vec![FieldId(0)]),
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document containing the exact phrase"
        );

        assert!(
            output.contains_key(&DocumentId(1)),
            "Document with ID 1 should be found"
        );
    }
}
