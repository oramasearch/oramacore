use std::collections::HashSet;

use filters::FilterResult;

use crate::{
    collection_manager::global_info::GlobalInfo,
    types::{DocumentId, FieldId},
};

pub struct FullTextSearchContext<'run, 'index> {
    pub tokens: &'run [String],
    pub exact_match: bool,
    pub boost: f32,
    pub field_id: FieldId,
    pub filtered_doc_ids: Option<&'run FilterResult<DocumentId>>,
    pub global_info: GlobalInfo,
    pub uncommitted_deleted_documents: &'index HashSet<DocumentId>,

    pub total_term_count: u64,
}

impl FullTextSearchContext<'_, '_> {
    pub fn increment_term_count(&mut self) {
        self.total_term_count += 1;
    }
}
