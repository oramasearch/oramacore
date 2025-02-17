// pub mod counter;
// pub mod gauge;
pub mod histogram;
use metrics::{Label, SharedString};

use crate::collection_manager::dto::FieldId;

pub struct Empty {}
impl From<Empty> for Vec<Label> {
    fn from(_: Empty) -> Self {
        vec![]
    }
}

macro_rules! create_label_struct {
    ($struct_name: tt, { $($field_name:ident : $field_type:ty),+ $(,)? }) => {
        pub struct $struct_name {
            $(pub $field_name: $field_type),+
        }

        impl From<$struct_name> for Vec<Label> {
            fn from(val: $struct_name) -> Self {
                vec![
                    $(Label::new(stringify!($field_name), val.$field_name)),+
                ]
            }
        }
    };
}

create_label_struct!(CollectionLabels, {
    collection: String,
});
create_label_struct!(EmbeddingCalculationLabels, {
    model: SharedString,
});
create_label_struct!(ChatCalculationLabels, {
    model: &'static str,
});
create_label_struct!(FieldCalculationLabels, {
    collection: SharedString,
    field: SharedString,
    field_type: SharedString,
});
create_label_struct!(CollectionCommitLabels, {
    collection: String,
    side: &'static str,
});
create_label_struct!(CollectionFieldCommitLabels, {
    collection: SharedString,
    field: FieldId,
    field_type: &'static str,
    side: &'static str,
});
create_label_struct!(SearchCollectionLabels, {
    collection: SharedString,
    mode: &'static str,
    has_filter: &'static str,
    has_facet: &'static str,
});
create_label_struct!(JSOperationLabels, {
    operation: &'static str,
});

pub mod ai {
    use super::{ChatCalculationLabels, EmbeddingCalculationLabels};
    use crate::{create_counter_histogram, create_time_histogram};
    create_time_histogram!(
        EMBEDDING_CALCULATION_TIME,
        "embedding_calculation_time_sec",
        EmbeddingCalculationLabels
    );
    create_counter_histogram!(
        EMBEDDING_CALCULATION_PARALLEL_COUNT,
        "embedding_calculation_batch_size",
        EmbeddingCalculationLabels
    );
    create_time_histogram!(
        CHAT_CALCULATION_TIME,
        "chat_calculation_time_sec",
        ChatCalculationLabels
    );
}

pub mod document_insertion {
    use super::{CollectionLabels, FieldCalculationLabels};
    use crate::create_time_histogram;
    create_time_histogram!(
        FIELD_CALCULATION_TIME,
        "writer_field_calculation_time_sec",
        FieldCalculationLabels
    );
    create_time_histogram!(
        DOCUMENT_CALCULATION_TIME,
        "writer_doc_calculation_time_sec",
        CollectionLabels
    );
}

pub mod commit {
    use super::{CollectionCommitLabels, CollectionFieldCommitLabels, Empty};
    use crate::create_time_histogram;
    create_time_histogram!(
        COMMIT_CALCULATION_TIME,
        "commit_calculation_time_sec",
        CollectionCommitLabels
    );
    create_time_histogram!(
        FIELD_COMMIT_CALCULATION_TIME,
        "field_commit_calculation_time_sec",
        CollectionFieldCommitLabels
    );
    create_time_histogram!(
        DOCUMENT_COMMIT_CALCULATION_TIME,
        "document_commit_calculation_time_sec",
        Empty
    );
}

pub mod operations {
    use super::CollectionLabels;
    use crate::create_counter_histogram;
    create_counter_histogram!(OPERATION_COUNT, "operation_count", CollectionLabels);
}

pub mod search {
    use super::{CollectionLabels, SearchCollectionLabels};
    use crate::{create_counter_histogram, create_time_histogram};
    create_time_histogram!(
        SEARCH_CALCULATION_TIME,
        "search_calculation_time_sec",
        SearchCollectionLabels
    );
    create_time_histogram!(
        FILTER_CALCULATION_TIME,
        "filter_calculation_time_sec",
        CollectionLabels
    );
    create_counter_histogram!(
        FILTER_PERC_CALCULATION_COUNT,
        "filter_percentage_calculation_count",
        CollectionLabels
    );
    create_counter_histogram!(
        FILTER_COUNT_CALCULATION_COUNT,
        "filter_count_calculation_count",
        CollectionLabels
    );
    create_counter_histogram!(
        MATCHING_PERC_CALCULATION_COUNT,
        "matching_percentage_calculation_count",
        CollectionLabels
    );
    create_counter_histogram!(
        MATCHING_COUNT_CALCULTATION_COUNT,
        "matching_calculation_count",
        CollectionLabels
    );
}

pub mod rabbit {
    use super::Empty;
    use crate::create_time_histogram;
    create_time_histogram!(
        RABBITMQ_ENQUEUE_CALCULATION_TIME,
        "rabbitmq_enqueue_calculation_time_sec",
        Empty
    );
}

pub mod js {
    use super::JSOperationLabels;
    use crate::create_time_histogram;
    create_time_histogram!(
        JS_CALCULATION_TIME,
        "js_calculation_time_sec",
        JSOperationLabels
    );
}

impl From<FieldId> for SharedString {
    fn from(val: FieldId) -> Self {
        SharedString::from(val.0.to_string())
    }
}

/*

pub static STRING_CALCULATION_METRIC: Histogram<StringCalculationLabels> = Histogram {
    key: "writer_calc_string_elapsed_sec",
    phantom: std::marker::PhantomData,
};

pub static SEARCH_METRIC: Histogram<SearchLabels> = Histogram {
    key: "reader_search_elapsed_sec",
    phantom: std::marker::PhantomData,
};

pub static SEARCH_FILTER_METRIC: Histogram<SearchFilterLabels> = Histogram {
    key: "reader_search_filter_elapsed_sec",
    phantom: std::marker::PhantomData,
};

pub static DOCUMENT_PROCESS_METRIC: Histogram<DocumentProcessLabels> = Histogram {
    key: "writer_doc_process_elapsed_sec",
    phantom: std::marker::PhantomData,
};

pub static COMMIT_METRIC: Histogram<CommitLabels> = Histogram {
    key: "commit_elapsed_sec",
    phantom: std::marker::PhantomData,
};
pub static SEARCH_FILTER_HISTOGRAM: Histogram<SearchFilterLabels> = Histogram {
    key: "reader_search_filter_matched_historgram",
    phantom: std::marker::PhantomData,
};


create_counter!(ADDED_DOCUMENTS_COUNTER, "writer_add_document_counter", AddedDocumentsLabels);
create_counter!(COLLECTION_ADDED_COUNTER, "reader_collection_added_counter", CollectionAddedLabels);
create_counter!(COLLECTION_OPERATION_COUNTER, "reader_collection_op_counter", CollectionOperationLabels);

create_gauge!(OPERATION_GAUGE, "operation_gauge", Empty);
create_gauge!(EMBEDDING_REQUEST_GAUDGE, "embedding_request_gauge", Empty);
create_gauge!(PENDING_EMBEDDING_REQUEST_GAUDGE, "pending_embedding_request_gauge", Empty);
create_gauge!(JAVASCRIPT_REQUEST_GAUDGE, "javascript_request_gauge", Empty);
*/
