pub mod histogram;
use metrics::{Label, SharedString};

use crate::types::FieldId;

pub struct Empty;
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
    field_type: SharedString,
});
create_label_struct!(CollectionCommitLabels, {
    collection: String,
    side: &'static str,
});
create_label_struct!(IndexCollectionCommitLabels, {
    collection: String,
    index: String,
    side: &'static str,
});
create_label_struct!(FieldIndexCollectionCommitLabels, {
    collection: String,
    index: String,
    field_type: &'static str,
    side: &'static str,
});
create_label_struct!(CollectionFieldCommitLabels, {
    collection: SharedString,
    field_type: &'static str,
    side: &'static str,
});
create_label_struct!(SearchCollectionLabels, {
    collection: SharedString,
    mode: &'static str,
    has_filter: &'static str,
    has_facet: &'static str,
    has_group: &'static str,
    has_sorting: &'static str,
});
create_label_struct!(JSOperationLabels, {
    operation: &'static str,
    collection: SharedString,
});

pub enum LockType {
    Read,
    Write,
    Mutex,
}
impl From<LockType> for SharedString {
    fn from(val: LockType) -> Self {
        match val {
            LockType::Read => SharedString::from("read"),
            LockType::Write => SharedString::from("write"),
            LockType::Mutex => SharedString::from("mutex"),
        }
    }
}
create_label_struct!(LockNameLabels, {
    name: &'static str,
    reason: &'static str,
    lock_type: LockType,
});

pub mod ai {
    use super::EmbeddingCalculationLabels;
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
    /*
    create_time_histogram!(
        CHAT_CALCULATION_TIME,
        "chat_calculation_time_sec",
        ChatCalculationLabels
    );
    create_time_histogram!(
        STREAM_CHAT_CALCULATION_TIME,
        "stream_chat_calculation_time_sec",
        ChatCalculationLabels
    );
    */
}

pub mod document_insertion {
    use super::{Empty, FieldCalculationLabels};
    use crate::create_time_histogram;
    create_time_histogram!(
        FIELD_CALCULATION_TIME,
        "writer_field_calculation_time_sec",
        FieldCalculationLabels
    );
    create_time_histogram!(
        DOCUMENTS_INSERTION_TIME,
        "writer_doc_calculation_time_sec",
        Empty
    );
}

pub mod commit {
    use super::Empty;
    use crate::{
        create_time_histogram,
        metrics::{
            CollectionCommitLabels, FieldIndexCollectionCommitLabels, IndexCollectionCommitLabels,
        },
    };
    create_time_histogram!(
        COMMIT_CALCULATION_TIME,
        "commit_calculation_time_sec",
        CollectionCommitLabels
    );
    create_time_histogram!(
        INDEX_COMMIT_CALCULATION_TIME,
        "index_commit_calculation_time_sec",
        IndexCollectionCommitLabels
    );
    create_time_histogram!(
        FIELD_INDEX_COMMIT_CALCULATION_TIME,
        "field_index_commit_calculation_time_sec",
        FieldIndexCollectionCommitLabels
    );
    create_time_histogram!(
        DOCUMENT_COMMIT_CALCULATION_TIME,
        "document_commit_calculation_time_sec",
        Empty
    );
}

pub mod operations {
    use super::Empty;
    use crate::create_time_histogram;
    create_time_histogram!(OPERATION_COUNT, "operation_count", Empty);
}

pub mod locks {
    use super::LockNameLabels;
    use crate::create_time_histogram;
    create_time_histogram!(LOCKING_TIME, "locking_time", LockNameLabels);
    create_time_histogram!(LOCKED_FOR_TIME, "locked_for_time", LockNameLabels);
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
