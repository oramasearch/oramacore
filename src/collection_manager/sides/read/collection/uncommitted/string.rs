use std::collections::{HashMap, HashSet};

use anyhow::Result;
use tracing::{debug, info, warn};

use crate::{
    collection_manager::{
        dto::{BM25Scorer, GlobalInfo},
        sides::{InsertStringTerms, TermStringField},
    },
    indexes::radix::RadixIndex,
    types::DocumentId,
};

/* The structure of data needed for BM25 scoring:
 * ```text
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
 * ```
 */

/// Total number of documents that contains a term in a field in the collection
#[derive(Debug, Clone)]
pub struct TotalDocumentsWithTermInField(pub u64);
impl TotalDocumentsWithTermInField {
    fn increment_by_one(&mut self) {
        self.0 += 1;
    }
}
#[derive(Debug, Clone)]
pub struct Positions(pub Vec<usize>);

#[derive(Debug)]
pub struct StringField {
    /// The sum of the length of all the content in the field in the collection
    total_field_length: usize,
    /// Set of document ids that has the field
    document_ids: HashSet<DocumentId>,
    /// The length for each document in the collection
    field_length_per_doc: HashMap<DocumentId, u32>,

    inner: RadixIndex<(
        TotalDocumentsWithTermInField,
        HashMap<DocumentId, Positions>,
    )>,
}

impl StringField {
    pub fn empty() -> Self {
        Self {
            total_field_length: 0,
            document_ids: HashSet::new(),
            field_length_per_doc: HashMap::new(),
            inner: RadixIndex::new(),
        }
    }

    pub fn global_info(&self) -> GlobalInfo {
        GlobalInfo {
            total_document_length: self.total_field_length,
            total_documents: self.document_ids.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.document_ids.len()
    }

    pub fn insert(&mut self, document_id: DocumentId, field_length: u16, terms: InsertStringTerms) {
        self.document_ids.insert(document_id);

        let max_position = terms
            .values()
            .flat_map(|term_string_field| term_string_field.positions.iter())
            .max()
            .unwrap_or(&0);
        self.field_length_per_doc
            .insert(document_id, *max_position as u32);

        for (term, term_string_field) in terms {
            let k = term.0;

            let TermStringField { positions } = term_string_field;

            self.total_field_length += usize::from(field_length);

            match self.inner.get_mut(k.bytes()) {
                Some(v) => {
                    v.0.increment_by_one();
                    let old_positions = v.1.entry(document_id).or_insert_with(|| Positions(vec![]));
                    old_positions.0.extend(positions);
                }
                None => {
                    self.inner.insert(
                        k.bytes(),
                        (
                            TotalDocumentsWithTermInField(1),
                            HashMap::from_iter([(document_id, Positions(positions))]),
                        ),
                    );
                }
            };
        }
    }

    pub fn field_length_per_doc(&self) -> HashMap<DocumentId, u32> {
        self.field_length_per_doc.clone()
    }

    pub fn search(
        &self,
        tokens: &[String],
        boost: f32,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        global_info: &GlobalInfo,
    ) -> Result<()> {
        let total_field_length = global_info.total_document_length as f32;
        let total_documents_with_field = global_info.total_documents as f32;
        let average_field_length = total_field_length / total_documents_with_field;

        let mut total_matches = 0_usize;
        for token in tokens {
            // We don't "boost" the exact match at all.
            // Should we boost if the match is "perfect"?
            // TODO: think about this
            let matches = self.inner.search(token)?;

            for (total_documents_with_term_in_field, position_per_document) in matches {
                for (doc_id, positions) in position_per_document {
                    if let Some(filtered_doc_ids) = filtered_doc_ids {
                        if !filtered_doc_ids.contains(doc_id) {
                            continue;
                        }
                    }

                    let field_length = match self.field_length_per_doc.get(doc_id) {
                        Some(field_length) => *field_length,
                        None => {
                            warn!("Document length not found for document_id: {:?}", doc_id);
                            continue;
                        }
                    };

                    let term_occurrence_in_field = positions.0.len() as u32;

                    // We aren't consider the phrase matching here
                    // Instead for committed data, we do.
                    // We should also here consider the phrase matching.
                    // TODO: Implement phrase matching

                    let total_documents_with_term_in_field =
                        total_documents_with_term_in_field.0 as usize;

                    scorer.add(
                        *doc_id,
                        term_occurrence_in_field,
                        field_length,
                        average_field_length,
                        global_info.total_documents as f32,
                        total_documents_with_term_in_field,
                        1.2,
                        0.75,
                        boost,
                    );

                    total_matches += 1;
                }
            }
        }

        debug!(total_matches = total_matches, "Uncommitted total matches");

        Ok(())
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<
        Item = (
            Vec<u8>,
            (
                TotalDocumentsWithTermInField,
                HashMap<DocumentId, Positions>,
            ),
        ),
    > + '_ {
        self.inner.inner.iter()
    }
}
