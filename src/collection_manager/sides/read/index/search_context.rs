use crate::{
    collection_manager::{global_info::GlobalInfo, sides::read::search::SearchDocumentContext},
    types::{DocumentId, FieldId},
};

pub struct FullTextSearchContext<'run, 'index> {
    pub tokens: &'run [String],
    pub exact_match: bool,
    pub boost: f32,
    pub field_id: FieldId,
    pub search_document_context: &'run SearchDocumentContext<'index, DocumentId>,
    pub global_info: GlobalInfo,

    pub total_term_count: u64,
}

impl FullTextSearchContext<'_, '_> {
    pub fn increment_term_count(&mut self) {
        self.total_term_count += 1;
    }
}
