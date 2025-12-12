use crate::types::DocumentList;

pub struct InsertDocuments {
    list: DocumentList,
}

impl InsertDocuments {
    pub fn new(list: DocumentList) -> Self {
        Self {
            list,
        }
    }

}

