use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
};

use anyhow::Result;
use dashmap::{DashMap, DashSet};
use deno_core::parking_lot::RwLock;
use ptrie::Trie;
use tracing::warn;

use crate::{
    collection_manager::sides::write::{InsertStringTerms, TermStringField},
    document_storage::DocumentId,
};

use super::scorer::BM25Scorer;

/**
 *
 * (coll_id, field_id) => {
 *    average_field_length: f32,
 *    total_documents_with_field: usize,
 *    
 *    (, doc_id) => {
 *      document_length
 *    }
 *
 *    (, term) => {
 *      total_documents_with_term_in_field: usize
 *      (, doc_id) => {
 *       term_occurrence_in_document: usize
 *       positions: Vec<u32>
 *      }
 *    }
 * }
 */

/// Total number of documents that contains a term in a field in the collection
#[derive(Debug, Clone)]
struct TotalDocumentsWithTermInField(u64);
impl TotalDocumentsWithTermInField {
    fn increment_by_one(&mut self) {
        self.0 += 1;
    }
}

/// Number of times a term occurs in a field in a document
#[derive(Debug, Clone)]
struct TermOccurrenceInDocument(u32);
impl TermOccurrenceInDocument {
    fn increment_by(&mut self, n: u32) {
        self.0 += n;
    }
}

#[derive(Debug, Clone)]
struct Positions(Vec<usize>);

#[derive(Debug)]
pub struct UncommittedStringFieldIndex {
    /// The sum of the length of all the content in the field in the collection
    total_field_length: AtomicU64,
    /// Set of document ids that has the field
    document_ids: DashSet<DocumentId>,
    /// The length for each document in the collection
    document_length_per_doc: DashMap<DocumentId, u32>,

    /// keep track of metrics for each term in the field
    inner: RwLock<
        Trie<
            u8,
            (
                TotalDocumentsWithTermInField,
                HashMap<DocumentId, (TermOccurrenceInDocument, Positions)>,
            ),
        >,
    >,

    id_generator: Arc<AtomicU32>,
}

impl UncommittedStringFieldIndex {
    pub fn new(id_generator: Arc<AtomicU32>) -> Self {
        Self {
            total_field_length: AtomicU64::new(0),
            document_ids: Default::default(),
            document_length_per_doc: Default::default(),
            inner: RwLock::new(Trie::new()),
            id_generator,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn insert(&self, document_id: DocumentId, terms: InsertStringTerms) -> Result<()> {
        self.document_ids.insert(document_id);

        let mut tree = self.inner.write();

        let max_position = terms
            .values()
            .flat_map(|term_string_field| term_string_field.positions.iter())
            .max()
            .unwrap_or(&0);
        self.document_length_per_doc
            .insert(document_id, *max_position as u32);

        for (term, term_string_field) in terms {
            let k = term.0;

            let TermStringField {
                field_length,
                positions,
            } = term_string_field;

            self.total_field_length
                .fetch_add(field_length.into(), Ordering::Relaxed);

            let occurrence = positions.len() as u32;

            match tree.get_mut(k.bytes()) {
                Some(v) => {
                    v.0.increment_by_one();
                    let v =
                        v.1.entry(document_id)
                            .or_insert_with(|| (TermOccurrenceInDocument(0), Positions(vec![])));
                    v.0.increment_by(occurrence);
                }
                None => {
                    tree.insert(
                        k.bytes(),
                        (
                            TotalDocumentsWithTermInField(1),
                            HashMap::from_iter([(
                                document_id,
                                (TermOccurrenceInDocument(occurrence), Positions(positions)),
                            )]),
                        ),
                    );
                }
            };
        }

        Ok(())
    }

    pub fn search(
        &self,
        tokens: &[String],
        boost: f32,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
    ) -> Result<()> {
        let p = self.inner.read();

        let total_field_length = self.total_field_length.load(Ordering::Relaxed) as f32;
        let total_documents_with_field = self.document_ids.len() as f32;
        let average_field_length = total_field_length / total_documents_with_field;
        /*
            /// The sum of the length of all the content in the field in the collection
        total_field_length: AtomicU64,
        /// The total number of documents that has that field in the collection
        total_documents_with_field: AtomicU64,
         */

        for token in tokens {
            let (current, mut postfixes) = p.find_postfixes_with_current(token.bytes());

            // We threat the current token as the postfixes
            // Should we boost if the match is "perfect"?
            // TODO: think about this

            if let Some(current) = current {
                postfixes.push(current);
            }

            for (total_documents_with_term_in_field, position_per_document) in postfixes {
                for (doc_id, (term_occurrence_in_document, _)) in position_per_document {
                    if let Some(filtered_doc_ids) = filtered_doc_ids {
                        if !filtered_doc_ids.contains(doc_id) {
                            continue;
                        }
                    }

                    let document_length = match self.document_length_per_doc.get(doc_id) {
                        Some(document_length) => *document_length,
                        None => {
                            warn!("Document length not found for document_id: {:?}", doc_id);
                            continue;
                        }
                    };

                    // We aren't consider the phrase matching here
                    // Instead for committed data, we do.
                    // We should also here consider the phrase matching.
                    // TODO: Implement phrase matching
                    scorer.add(
                        *doc_id,
                        term_occurrence_in_document.0,
                        document_length,
                        average_field_length,
                        total_documents_with_field,
                        total_documents_with_term_in_field.0 as usize,
                        1.2,
                        0.75,
                        boost,
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::collection_manager::sides::write::FieldIndexer;
    use crate::{
        collection_manager::{
            dto::FieldId,
            sides::write::{CollectionWriteOperation, StringField, WriteOperation},
            CollectionId,
        },
        indexes::string::scorer::BM25Scorer,
        nlp::TextParser,
        types::Document,
    };

    use super::*;

    fn create_uncommitted_string_field_index(documents: Vec<Document>) -> Result<UncommittedStringFieldIndex> {
        let index = UncommittedStringFieldIndex::new(Arc::new(AtomicU32::new(0)));

        let string_field = StringField::new(Arc::new(TextParser::from_language(
            crate::nlp::locales::Locale::EN,
        )));

        for (id, doc) in documents.into_iter().enumerate() {
            let document_id = DocumentId(id as u32);
            let flatten = doc.into_flatten();
            let operations = string_field.get_write_operations(
                CollectionId("collection".to_string()),
                document_id,
                "field",
                FieldId(1),
                &flatten,
            )?;

            for operation in operations {
                match operation {
                    WriteOperation::Collection(
                        _,
                        CollectionWriteOperation::IndexString { terms, .. },
                    ) => {
                        index.insert(document_id, terms)?;
                    }
                    _ => unreachable!(),
                };
            }
        }

        Ok(index)
    }

    #[test]
    fn test_indexes_string_uncommitted() -> Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let index = create_uncommitted_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])?;

        // Exact match
        let mut scorer = BM25Scorer::new();
        index.search(&["hello".to_string()], 1.0, &mut scorer, None)?;
        let exact_match_output = scorer.get_scores();
        assert_eq!(
            exact_match_output.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from_iter([DocumentId(0), DocumentId(1)])
        );
        assert!(exact_match_output[&DocumentId(0)] > exact_match_output[&DocumentId(1)]);

        // Prefix match
        let mut scorer = BM25Scorer::new();
        index.search(&["hel".to_string()], 1.0, &mut scorer, None)?;
        let prefix_match_output = scorer.get_scores();
        assert_eq!(
            prefix_match_output.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from_iter([DocumentId(0), DocumentId(1)])
        );

        Ok(())
    }

    #[test]
    fn test_indexes_string_uncommitted_boost() -> Result<()> {
        let index = create_uncommitted_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])?;

        // 1.0
        let mut scorer = BM25Scorer::new();
        index.search(&["hello".to_string()], 1.0, &mut scorer, None)?;
        let base_output = scorer.get_scores();

        // 0.5
        let mut scorer = BM25Scorer::new();
        index.search(&["hello".to_string()], 0.5, &mut scorer, None)?;
        let half_boost_output = scorer.get_scores();

        // 2.0
        let mut scorer = BM25Scorer::new();
        index.search(&["hello".to_string()], 2.0, &mut scorer, None)?;
        let twice_boost_output = scorer.get_scores();

        assert!(base_output[&DocumentId(0)] > half_boost_output[&DocumentId(0)]);
        assert!(base_output[&DocumentId(0)] < twice_boost_output[&DocumentId(0)]);

        assert!(base_output[&DocumentId(1)] > half_boost_output[&DocumentId(1)]);
        assert!(base_output[&DocumentId(1)] < twice_boost_output[&DocumentId(1)]);

        Ok(())
    }


    #[test]
    fn test_indexes_string_uncommitted_nonexistent_term() -> Result<()> {
        let index = create_uncommitted_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])?;

        let mut scorer = BM25Scorer::new();
        index
            .search(
                &["nonexistent".to_string()],
                1.0,
                &mut scorer,
                None,
            )?;
        let output = scorer.get_scores();

        assert!(
            output.is_empty(),
            "Search results should be empty for non-existent term"
        );

        Ok(())
    }


    #[test]
    fn test_indexes_string_uncommitted_field_filter() -> Result<()> {
        let index = create_uncommitted_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])?;

        // Exclude a doc
        {
            let mut scorer = BM25Scorer::new();
            index
                .search(
                    &["hello".to_string()],
                    1.0,
                    &mut scorer,
                    Some(&HashSet::from_iter([DocumentId(0)])),
                )?;
            let output = scorer.get_scores();
            assert!(
                output.contains_key(&DocumentId(0)),
            );
            assert!(
                !output.contains_key(&DocumentId(1)),
            );
        }

        // Exclude all docs
        {
            let mut scorer = BM25Scorer::new();
            index
                .search(
                    &["hello".to_string()],
                    1.0,
                    &mut scorer,
                    Some(&HashSet::new()),
                )?;
            let output = scorer.get_scores();
            assert!(
                output.is_empty(),
            );
        }

        Ok(())
    }

    #[test]
    fn test_indexes_string_uncommitted_on_empty_index() -> Result<()>{
        let index = create_uncommitted_string_field_index(vec![])?;

        let mut scorer = BM25Scorer::new();
        index
            .search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                None,
            )?;
        let output = scorer.get_scores();
        assert!(
            output.is_empty(),
        );

        Ok(())
    }


    #[test]
    fn test_indexes_string_uncommitted_large_text() -> Result<()> {
        let index = create_uncommitted_string_field_index(vec![
            json!({
                "field": "word ".repeat(10000),
            }).try_into()?,
        ])?;

        
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &["word".to_string()],
                1.0,
                &mut scorer,
                None,
            )?;
        let output = scorer.get_scores();
        assert_eq!(
            output.len(),
            1,
            "Should find the document containing the large text"
        );

        Ok(())
    }
}
