use std::{
    collections::{HashMap, HashSet},
    sync::RwLock,
};

use anyhow::Result;
use ptrie::Trie;
use tracing::{info, warn};

use crate::{
    collection_manager::sides::{InsertStringTerms, TermStringField},
    types::DocumentId,
};

use super::{scorer::BM25Scorer, GlobalInfo};

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

#[derive(Debug, Default, Clone)]
pub struct InnerInnerUncommittedStringFieldIndex {
    /// The sum of the length of all the content in the field in the collection
    pub total_field_length: u64,
    /// Set of document ids that has the field
    pub document_ids: HashSet<DocumentId>,
    /// The length for each document in the collection
    pub field_length_per_doc: HashMap<DocumentId, u32>,

    /// keep track of metrics for each term in the field
    pub tree: Trie<
        u8,
        (
            TotalDocumentsWithTermInField,
            HashMap<DocumentId, Positions>,
        ),
    >,
}

impl InnerInnerUncommittedStringFieldIndex {
    pub fn new() -> Self {
        Self {
            total_field_length: 0,
            document_ids: Default::default(),
            field_length_per_doc: Default::default(),
            tree: Trie::new(),
        }
    }

    pub fn insert(
        &mut self,
        document_id: DocumentId,
        field_length: u16,
        terms: InsertStringTerms,
    ) -> Result<()> {
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

            self.total_field_length += u64::from(field_length);

            match self.tree.get_mut(k.bytes()) {
                Some(v) => {
                    v.0.increment_by_one();
                    let old_positions = v.1.entry(document_id).or_insert_with(|| Positions(vec![]));
                    old_positions.0.extend(positions);
                }
                None => {
                    self.tree.insert(
                        k.bytes(),
                        (
                            TotalDocumentsWithTermInField(1),
                            HashMap::from_iter([(document_id, Positions(positions))]),
                        ),
                    );
                }
            };
        }

        Ok(())
    }

    pub fn get_global_info(&self) -> GlobalInfo {
        GlobalInfo {
            total_document_length: self.total_field_length as usize,
            total_documents: self.document_ids.len(),
        }
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
            let (current, mut postfixes) = self.tree.find_postfixes_with_current(token.bytes());

            // We don't "boost" the exact match at all.
            // Should we boost if the match is "perfect"?
            // TODO: think about this

            if let Some(current) = current {
                postfixes.push(current);
            }

            for (total_documents_with_term_in_field, position_per_document) in postfixes {
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

        info!(total_matches = total_matches, "Uncommitted total matches");

        Ok(())
    }

    fn clear(&mut self) {
        self.total_field_length = 0;
        self.document_ids.clear();
        self.field_length_per_doc.clear();
        self.tree.clear();
    }
}

#[derive(Debug, Default)]
pub struct InnerUncommittedStringFieldIndex {
    left: InnerInnerUncommittedStringFieldIndex,
    right: InnerInnerUncommittedStringFieldIndex,
    state: bool,
}

impl InnerUncommittedStringFieldIndex {
    pub fn new() -> Self {
        Self {
            left: InnerInnerUncommittedStringFieldIndex::new(),
            right: InnerInnerUncommittedStringFieldIndex::new(),
            state: false,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn insert(
        &mut self,
        document_id: DocumentId,
        field_length: u16,
        terms: InsertStringTerms,
    ) -> Result<()> {
        let inner = if self.state {
            &mut self.left
        } else {
            &mut self.right
        };

        inner.insert(document_id, field_length, terms)
    }

    pub fn get_global_info(&self) -> GlobalInfo {
        self.left.get_global_info() + self.right.get_global_info()
    }

    pub fn search(
        &self,
        tokens: &[String],
        boost: f32,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        global_info: &GlobalInfo,
    ) -> Result<()> {
        self.left
            .search(tokens, boost, scorer, filtered_doc_ids, global_info)?;
        self.right
            .search(tokens, boost, scorer, filtered_doc_ids, global_info)?;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct UncommittedStringFieldIndex {
    inner: RwLock<InnerUncommittedStringFieldIndex>,
}

impl UncommittedStringFieldIndex {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(InnerUncommittedStringFieldIndex::new()),
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn insert(
        &self,
        document_id: DocumentId,
        field_length: u16,
        terms: InsertStringTerms,
    ) -> Result<()> {
        let mut inner = match self.inner.write() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };
        inner.insert(document_id, field_length, terms)
    }

    pub fn get_global_info(&self) -> GlobalInfo {
        let inner = match self.inner.read() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };
        inner.get_global_info()
    }

    pub fn search(
        &self,
        tokens: &[String],
        boost: f32,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        global_info: &GlobalInfo,
    ) -> Result<()> {
        let inner = match self.inner.read() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };
        inner.search(tokens, boost, scorer, filtered_doc_ids, global_info)
    }

    pub fn take(&self) -> Result<DataToCommit> {
        let mut inner = match self.inner.write() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };
        let data = if inner.state {
            inner.left.clone()
        } else {
            inner.right.clone()
        };
        let current_state = inner.state;

        // Route the writes to the other side
        inner.state = !current_state;
        drop(inner);

        let InnerInnerUncommittedStringFieldIndex {
            total_field_length,
            document_ids,
            field_length_per_doc,
            tree,
        } = data;

        let mut tree: Vec<_> = tree.into_iter().map(|(k, v)| (k.to_vec(), v)).collect();
        tree.sort_by(|(a, _), (b, _)| a.cmp(b));

        Ok(DataToCommit {
            index: self,
            state_to_clear: current_state,
            total_field_length,
            document_ids,
            field_length_per_doc,
            tree,
        })
    }
}

pub struct DataToCommit<'uncommitted> {
    index: &'uncommitted UncommittedStringFieldIndex,
    state_to_clear: bool,

    // the state
    total_field_length: u64,
    document_ids: HashSet<DocumentId>,
    field_length_per_doc: HashMap<DocumentId, u32>,
    #[allow(clippy::type_complexity)]
    tree: Vec<(
        Vec<u8>,
        (
            TotalDocumentsWithTermInField,
            HashMap<DocumentId, Positions>,
        ),
    )>,
}

impl DataToCommit<'_> {
    pub fn get_document_lengths(&self) -> &HashMap<DocumentId, u32> {
        &self.field_length_per_doc
    }

    pub fn global_info(&self) -> GlobalInfo {
        GlobalInfo {
            total_document_length: self.total_field_length as usize,
            total_documents: self.document_ids.len(),
        }
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
        self.tree.iter().map(|(k, v)| (k.to_vec(), v.clone()))
    }

    // Invoke drop to clear the data
    pub fn done(self) {}
}

impl Drop for DataToCommit<'_> {
    fn drop(&mut self) {
        let mut lock = self.index.inner.write().unwrap();
        let tree = if self.state_to_clear {
            &mut lock.left
        } else {
            &mut lock.right
        };
        tree.clear();
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::collection_manager::sides::Term;
    use crate::indexes::string::scorer::BM25Scorer;
    use crate::test_utils::create_uncommitted_string_field_index;

    use super::*;

    #[tokio::test]
    async fn test_indexes_string_uncommitted() -> Result<()> {
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
        ])
        .await?;

        // Exact match
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let exact_match_output = scorer.get_scores();
        assert_eq!(
            exact_match_output.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from_iter([DocumentId(0), DocumentId(1)])
        );
        assert!(exact_match_output[&DocumentId(0)] > exact_match_output[&DocumentId(1)]);

        // Prefix match
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hel".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let prefix_match_output = scorer.get_scores();
        assert_eq!(
            prefix_match_output.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from_iter([DocumentId(0), DocumentId(1)])
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_indexes_string_uncommitted_boost() -> Result<()> {
        let index = create_uncommitted_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])
        .await?;

        // 1.0
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let base_output = scorer.get_scores();

        // 0.5
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            0.5,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let half_boost_output = scorer.get_scores();

        // 2.0
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            2.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let twice_boost_output = scorer.get_scores();

        assert!(base_output[&DocumentId(0)] > half_boost_output[&DocumentId(0)]);
        assert!(base_output[&DocumentId(0)] < twice_boost_output[&DocumentId(0)]);

        assert!(base_output[&DocumentId(1)] > half_boost_output[&DocumentId(1)]);
        assert!(base_output[&DocumentId(1)] < twice_boost_output[&DocumentId(1)]);

        Ok(())
    }

    #[tokio::test]
    async fn test_indexes_string_uncommitted_nonexistent_term() -> Result<()> {
        let index = create_uncommitted_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])
        .await?;

        let mut scorer = BM25Scorer::new();
        index.search(
            &["nonexistent".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let output = scorer.get_scores();

        assert!(
            output.is_empty(),
            "Search results should be empty for non-existent term"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_indexes_string_uncommitted_field_filter() -> Result<()> {
        let index = create_uncommitted_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])
        .await?;

        // Exclude a doc
        {
            let mut scorer = BM25Scorer::new();
            index.search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                Some(&HashSet::from_iter([DocumentId(0)])),
                &index.get_global_info(),
            )?;
            let output = scorer.get_scores();
            assert!(output.contains_key(&DocumentId(0)),);
            assert!(!output.contains_key(&DocumentId(1)),);
        }

        // Exclude all docs
        {
            let mut scorer = BM25Scorer::new();
            index.search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                Some(&HashSet::new()),
                &index.get_global_info(),
            )?;
            let output = scorer.get_scores();
            assert!(output.is_empty(),);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_indexes_string_uncommitted_on_empty_index() -> Result<()> {
        let index = create_uncommitted_string_field_index(vec![]).await?;

        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let output = scorer.get_scores();
        assert!(output.is_empty(),);

        Ok(())
    }

    #[tokio::test]
    async fn test_indexes_string_uncommitted_large_text() -> Result<()> {
        let index = create_uncommitted_string_field_index(vec![json!({
            "field": "word ".repeat(10000),
        })
        .try_into()?])
        .await?;

        let mut scorer = BM25Scorer::new();
        index.search(
            &["word".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let output = scorer.get_scores();
        assert_eq!(
            output.len(),
            1,
            "Should find the document containing the large text"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_indexes_string_uncommitted_during_commit() -> Result<()> {
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
        ])
        .await?;

        // Exact match
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let first_match_output = scorer.get_scores();
        assert_eq!(
            first_match_output.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from_iter([DocumentId(0), DocumentId(1)])
        );
        assert!(first_match_output[&DocumentId(0)] > first_match_output[&DocumentId(1)]);

        // During the commit phase
        let data_to_commit = index.take()?;

        // The previous data should still be available
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let second_match_output = scorer.get_scores();
        assert_eq!(first_match_output, second_match_output);

        // Insertion is still possible
        index.insert(
            DocumentId(2),
            1,
            HashMap::from_iter([(
                Term("hello".to_string()),
                TermStringField { positions: vec![0] },
            )]),
        )?;

        // And the results are combined
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let third_match_output = scorer.get_scores();
        assert!(third_match_output.contains_key(&DocumentId(0)));
        assert!(third_match_output.contains_key(&DocumentId(1)));
        assert!(third_match_output.contains_key(&DocumentId(2)));

        data_to_commit.done();

        // After the commit, only the new data is available
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let fourth_match_output = scorer.get_scores();
        assert!(!fourth_match_output.contains_key(&DocumentId(0)));
        assert!(!fourth_match_output.contains_key(&DocumentId(1)));
        assert!(fourth_match_output.contains_key(&DocumentId(2)));

        Ok(())
    }
}
