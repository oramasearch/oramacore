pub mod histogram;
use metrics::{Label, SharedString};

use crate::{
    create_counter, create_time_histogram,
    types::{CollectionId, FieldId},
};

// Define counter metric for insert_documents method invocations
create_counter!(
    INSERT_DOCUMENTS_COUNTER,
    "oramacore_insert_documents_total",
    CollectionIdLabel
);

// Define time histogram metric for insert_documents duration
create_time_histogram!(
    INSERT_DOCUMENTS_DURATION,
    "oramacore_insert_documents_duration_seconds",
    CollectionIdLabel
);

// Define counter metric for update_documents method invocations
create_counter!(
    UPDATE_DOCUMENTS_COUNTER,
    "oramacore_update_documents_total",
    CollectionIdLabel
);

// Define time histogram metric for update_documents duration
create_time_histogram!(
    UPDATE_DOCUMENTS_DURATION,
    "oramacore_update_documents_duration_seconds",
    CollectionIdLabel
);

// Define counter metric for delete_documents method invocations
create_counter!(
    DELETE_DOCUMENTS_COUNTER,
    "oramacore_delete_documents_total",
    CollectionIdLabel
);

// Define time histogram metric for delete_documents duration
create_time_histogram!(
    DELETE_DOCUMENTS_DURATION,
    "oramacore_delete_documents_duration_seconds",
    CollectionIdLabel
);

pub struct Empty;
impl From<Empty> for Vec<Label> {
    fn from(_: Empty) -> Self {
        vec![]
    }
}

pub struct CollectionIdLabel {
    pub collection_id: CollectionId,
}

impl From<CollectionIdLabel> for Vec<Label> {
    fn from(label: CollectionIdLabel) -> Self {
        vec![Label::new("collection_id", label.collection_id)]
    }
}

impl From<FieldId> for SharedString {
    fn from(val: FieldId) -> Self {
        SharedString::from(val.0.to_string())
    }
}

impl From<CollectionId> for SharedString {
    fn from(val: CollectionId) -> Self {
        SharedString::from(val.to_string())
    }
}
