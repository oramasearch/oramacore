use std::ops::{Add, AddAssign};

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GlobalInfo {
    pub total_documents: usize,
    pub total_document_length: usize,
}
impl AddAssign for GlobalInfo {
    fn add_assign(&mut self, rhs: Self) {
        self.total_documents += rhs.total_documents;
        self.total_document_length += rhs.total_document_length;
    }
}
impl Add for GlobalInfo {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            total_documents: self.total_documents + rhs.total_documents,
            total_document_length: self.total_document_length + rhs.total_document_length,
        }
    }
}
