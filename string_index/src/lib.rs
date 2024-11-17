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
        let temp_map = self.temp_map.read().unwrap();

        let mut entries: Vec<_> = temp_map
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    ((value.posting_list_id.0 as u64) << 32) | (value.term_frequency as u64),
                )
            })
            .collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        let mut builder = MapBuilder::memory();
        for (key, value) in entries {
            builder.insert(key, value)?;
        }

        Ok(builder.into_map())
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

        let mut posting_list_ids_with_freq = Vec::new();

        if let Some(fst) = self.fst_map.read().unwrap().as_ref() {
            for token in tokens {
                let automaton = fst::automaton::Str::new(&token).starts_with();
                let mut stream = fst.search(automaton).into_stream();

                while let Some(result) = stream.next() {
                    let (_, packed_value) = result;
                    let posting_list_id = PostingListId((packed_value >> 32).try_into().unwrap());
                    let term_frequency = packed_value & 0xFFFFFFFF;

                    posting_list_ids_with_freq.push(StringIndexValue {
                        posting_list_id: PostingListId(posting_list_id.0),
                        term_frequency: term_frequency as usize,
                    });
                }
            }
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
            .for_each(|(postings, total_token_count)| {
                let total_token_count = total_token_count as f32;
                for posting in postings {
                    let boost_per_field = *boost.get(&posting.field_id).unwrap_or(&1.0);
                    scorer.add_entry(&global_info, posting, total_token_count, boost_per_field);
                }
            });

        Ok(scorer.get_scores())
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

        let new_fst = self.build_fst()?;
        *self.fst_map.write().unwrap() = Some(new_fst);

        postings_per_posting_list_id
            .into_par_iter()
            .map(|(k, v)| self.posting_storage.add_or_create(k, v))
            .all(|_| true);

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
