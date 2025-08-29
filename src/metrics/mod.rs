pub mod histogram;
use metrics::{Label, SharedString};

use crate::{
    create_counter, create_time_histogram,
    types::{CollectionId, FieldId},
};

const COLLECTION_ID_LABEL_KEY: &str = "collection_id";

create_counter!(
    INSERT_DOCUMENTS_COUNTER,
    "oramacore_insert_documents_total",
    CollectionIdLabel
);

create_time_histogram!(
    INSERT_DOCUMENTS_DURATION,
    "oramacore_insert_documents_duration_seconds",
    CollectionIdLabel
);

create_counter!(
    UPDATE_DOCUMENTS_COUNTER,
    "oramacore_update_documents_total",
    CollectionIdLabel
);

create_time_histogram!(
    UPDATE_DOCUMENTS_DURATION,
    "oramacore_update_documents_duration_seconds",
    CollectionIdLabel
);

create_counter!(
    DELETE_DOCUMENTS_COUNTER,
    "oramacore_delete_documents_total",
    CollectionIdLabel
);

create_time_histogram!(
    DELETE_DOCUMENTS_DURATION,
    "oramacore_delete_documents_duration_seconds",
    CollectionIdLabel
);

create_counter!(SEARCH_COUNTER, "oramacore_search_total", SearchMetricsLabel);

create_time_histogram!(
    SEARCH_DURATION,
    "oramacore_search_duration_seconds",
    SearchMetricsLabel
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
        vec![Label::new(COLLECTION_ID_LABEL_KEY, label.collection_id)]
    }
}

pub struct SearchMetricsLabel {
    pub collection_id: CollectionId,
    pub has_filter: bool,
    pub has_facets: bool,
}

impl From<SearchMetricsLabel> for Vec<Label> {
    fn from(label: SearchMetricsLabel) -> Self {
        vec![
            Label::new(COLLECTION_ID_LABEL_KEY, label.collection_id),
            Label::new(
                "has_filter",
                if label.has_filter { "true" } else { "false" },
            ),
            Label::new(
                "has_facets",
                if label.has_facets { "true" } else { "false" },
            ),
        ]
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
