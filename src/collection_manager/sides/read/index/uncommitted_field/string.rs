use std::collections::{HashMap, HashSet};

use anyhow::Result;
use serde::Serialize;
use tracing::{debug, warn};

use crate::{
    collection_manager::{
        bm25::BM25Scorer,
        global_info::GlobalInfo,
        sides::{
            read::index::search_context::FullTextSearchContext, InsertStringTerms, TermStringField,
        },
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
pub struct UncommittedStringField {
    field_path: Box<[String]>,

    /// The sum of the length of all the content in the field in the collection
    total_field_length: usize,
    /// Set of document ids that has the field
    document_ids: HashSet<DocumentId>,
    /// The length for each document in the collection
    field_length_per_doc: HashMap<DocumentId, u32>,

    inner: RadixIndex<(
        TotalDocumentsWithTermInField,
        // doc_id => (exact positions, stemmed positions)
        HashMap<DocumentId, (Positions, Positions)>,
    )>,
}

impl UncommittedStringField {
    pub fn empty(field_path: Box<[String]>) -> Self {
        Self {
            field_path,
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

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn len(&self) -> usize {
        self.document_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.inner = RadixIndex::new();
        self.document_ids = HashSet::new();
        self.field_length_per_doc = HashMap::new();
        self.total_field_length = 0;
    }

    pub fn insert(&mut self, document_id: DocumentId, field_length: u16, terms: InsertStringTerms) {
        self.document_ids.insert(document_id);

        let max_position = terms
            .values()
            .flat_map(|term_string_field| term_string_field.positions.iter())
            .max()
            .unwrap_or(&0);
        // NB: the position is 0 based, so we need to add 1
        let document_length = *max_position as u32 + 1;

        self.field_length_per_doc
            .insert(document_id, document_length);

        self.total_field_length += usize::from(field_length);
        for (term, term_string_field) in terms {
            let k = term.0;

            let TermStringField {
                positions,
                exact_positions,
            } = term_string_field;

            match self.inner.get_mut(k.bytes()) {
                Some(v) => {
                    v.0.increment_by_one();
                    let old_positions =
                        v.1.entry(document_id)
                            .or_insert_with(|| (Positions(vec![]), Positions(vec![])));
                    old_positions.0 .0.extend(exact_positions);
                    old_positions.1 .0.extend(positions);
                }
                None => {
                    self.inner.insert(
                        k.bytes(),
                        (
                            TotalDocumentsWithTermInField(1),
                            HashMap::from_iter([(
                                document_id,
                                (Positions(exact_positions), Positions(positions)),
                            )]),
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
        context: &mut FullTextSearchContext<'_, '_>,
        scorer: &mut BM25Scorer<DocumentId>,
    ) -> Result<()> {
        let total_field_length = context.global_info.total_document_length as f32;
        let total_documents_with_field = context.global_info.total_documents as f32;
        let average_field_length = total_field_length / total_documents_with_field;

        let mut total_matches = 0_usize;
        for token in context.tokens {
            scorer.next_term();

            context.increment_term_count();

            // We don't "boost" the exact match at all.
            // Should we boost if the match is "perfect"?
            // TODO: think about this
            let matches = if context.exact_match {
                self.inner.search_exact(token)?
            } else {
                self.inner.search(token)?
            };

            for (total_documents_with_term_in_field, position_per_document) in matches {
                for (doc_id, positions) in position_per_document {
                    if let Some(filtered_doc_ids) = context.filtered_doc_ids {
                        if !filtered_doc_ids.contains(doc_id) {
                            continue;
                        }
                    }
                    if context.uncommitted_deleted_documents.contains(doc_id) {
                        continue;
                    }

                    let field_length = match self.field_length_per_doc.get(doc_id) {
                        Some(field_length) => *field_length,
                        None => {
                            warn!("Document length not found for document_id: {:?}", doc_id);
                            continue;
                        }
                    };

                    let term_occurrence_in_field = if context.exact_match {
                        if positions.0 .0.is_empty() {
                            continue;
                        }
                        positions.0 .0.len() as u32
                    } else {
                        (positions.1 .0.len() + positions.0 .0.len()) as u32
                    };

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
                        context.global_info.total_documents as f32,
                        total_documents_with_term_in_field,
                        1.2,
                        0.75,
                        context.boost,
                        0,
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
                HashMap<DocumentId, (Positions, Positions)>,
            ),
        ),
    > + '_ {
        self.inner.inner.iter()
    }

    pub fn stats(&self) -> UncommittedStringFieldStats {
        UncommittedStringFieldStats {
            key_count: self.inner.len(),
            global_info: self.global_info(),
        }
    }
}

#[derive(Serialize, Debug)]
pub struct UncommittedStringFieldStats {
    pub key_count: usize,
    pub global_info: GlobalInfo,
}
