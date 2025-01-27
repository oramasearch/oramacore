use std::time::Instant;

use metrics::{counter, gauge, histogram, IntoF64, Label};
use num_traits::cast;
use tracing::error;

pub struct HistogramKey {
    key: &'static str,
}

macro_rules! create_label_struct {
    ($struct_name:ident, { $($field_name:ident : $field_type:ty),+ $(,)? }) => {
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
create_label_struct!(EmbeddingCalculationLabels, {
    model: String,
});
pub static EMBEDDING_CALCULATION_METRIC: DeltaMetric<EmbeddingCalculationLabels> = DeltaMetric {
    elapsed: &HistogramKey {
        key: "writer_calc_embed_elapsed_sec",
    },
    phantom: std::marker::PhantomData,
};
create_label_struct!(StringCalculationLabels, {
    collection: String,
    field: String,
});
pub static STRING_CALCULATION_METRIC: DeltaMetric<StringCalculationLabels> = DeltaMetric {
    elapsed: &HistogramKey {
        key: "writer_calc_string_elapsed_sec",
    },
    phantom: std::marker::PhantomData,
};
create_label_struct!(SearchLabels, {
    collection: String,
});
pub static SEARCH_METRIC: DeltaMetric<SearchLabels> = DeltaMetric {
    elapsed: &HistogramKey {
        key: "reader_search_elapsed_sec",
    },
    phantom: std::marker::PhantomData,
};
create_label_struct!(SearchFilterLabels, {
    collection: String,
});
pub static SEARCH_FILTER_METRIC: DeltaMetric<SearchFilterLabels> = DeltaMetric {
    elapsed: &HistogramKey {
        key: "reader_search_filter_elapsed_sec",
    },
    phantom: std::marker::PhantomData,
};
create_label_struct!(DocumentProcessLabels, {
    collection: String,
});
pub static DOCUMENT_PROCESS_METRIC: DeltaMetric<DocumentProcessLabels> = DeltaMetric {
    elapsed: &HistogramKey {
        key: "writer_doc_process_elapsed_sec",
    },
    phantom: std::marker::PhantomData,
};
create_label_struct!(CommitLabels, {
    side: &'static str,
    collection: String,
    index_type: &'static str,
});
pub static COMMIT_METRIC: DeltaMetric<CommitLabels> = DeltaMetric {
    elapsed: &HistogramKey {
        key: "commit_elapsed_sec",
    },
    phantom: std::marker::PhantomData,
};

pub struct DeltaMetric<L> {
    elapsed: &'static HistogramKey,
    phantom: std::marker::PhantomData<L>,
}

impl<L: Into<Vec<Label>>> DeltaMetric<L> {
    pub fn create(&self, labels: L) -> DeltaMetricImpl {
        DeltaMetricImpl {
            elapsed: self.elapsed,
            now: Instant::now(),
            labels: labels.into(),
        }
    }
}

pub struct DeltaMetricImpl {
    elapsed: &'static HistogramKey,
    now: Instant,
    labels: Vec<Label>,
}

impl Drop for DeltaMetricImpl {
    fn drop(&mut self) {
        let elapsed = self.now.elapsed();
        let histogram = histogram!(self.elapsed.key, self.labels.clone());
        histogram.record(elapsed.as_secs_f64());
    }
}

pub struct Counter<Labels> {
    key: &'static str,
    phantom: std::marker::PhantomData<Labels>,
}
impl<Labels: Into<Vec<Label>>> Counter<Labels> {
    pub fn create(&self, labels: Labels) -> CounterImpl {
        CounterImpl {
            key: self.key,
            labels: labels.into(),
        }
    }
}
pub struct CounterImpl {
    key: &'static str,
    labels: Vec<Label>,
}
impl CounterImpl {
    pub fn increment_by_one(self) {
        counter!(self.key, self.labels).increment(1);
    }
    pub fn increment_by(self, value: usize) {
        if let Some(value) = cast::<usize, u64>(value) {
            let counter = counter!(self.key, self.labels);
            counter.increment(value);
        } else {
            error!("Failed to cast value to i64");
        }
    }
}

create_label_struct!(AddedDocumentsLabels, {
    collection: String,
});
pub static ADDED_DOCUMENTS_COUNTER: Counter<AddedDocumentsLabels> = Counter {
    key: "writer_add_document_counter",
    phantom: std::marker::PhantomData,
};
create_label_struct!(CollectionAddedLabels, {
    collection: String,
});
pub static COLLECTION_ADDED_COUNTER: Counter<CollectionAddedLabels> = Counter {
    key: "reader_collection_added_counter",
    phantom: std::marker::PhantomData,
};
create_label_struct!(CollectionOperationLabels, {
    collection: String,
});
pub static COLLECTION_OPERATION_COUNTER: Counter<CollectionOperationLabels> = Counter {
    key: "reader_collection_op_counter",
    phantom: std::marker::PhantomData,
};

pub struct Histogram<Labels> {
    key: &'static str,
    phantom: std::marker::PhantomData<Labels>,
}
impl<Labels: Into<Vec<Label>>> Histogram<Labels> {
    pub fn create(&self, labels: Labels) -> HistogramImpl {
        HistogramImpl {
            key: self.key,
            labels: labels.into(),
        }
    }
}
pub struct HistogramImpl {
    key: &'static str,
    labels: Vec<Label>,
}
impl HistogramImpl {
    pub fn record(self, value: f64) {
        histogram!(self.key, self.labels).record(value);
    }
    pub fn record_usize(self, value: usize) {
        if let Some(value) = cast::<usize, f64>(value) {
            self.record(value);
        } else {
            error!("Failed to cast value to f64");
        }
    }
}
pub static SEARCH_FILTER_HISTOGRAM: Histogram<SearchFilterLabels> = Histogram {
    key: "reader_search_filter_matched_historgram",
    phantom: std::marker::PhantomData,
};

pub struct Gauge<Labels> {
    key: &'static str,
    phantom: std::marker::PhantomData<Labels>,
}
impl<Labels: Into<Vec<Label>>> Gauge<Labels> {
    pub fn create(&self, labels: Labels) -> GaugeImpl {
        GaugeImpl {
            key: self.key,
            labels: labels.into(),
        }
    }
}
pub struct GaugeImpl {
    key: &'static str,
    labels: Vec<Label>,
}
impl GaugeImpl {
    pub fn increment_by_one(self) {
        self.increment_by(1);
    }
    pub fn decrement_by_one(self) {
        self.decrement_by(1);
    }
    pub fn increment_by<T: IntoF64>(self, value: T) {
        let gauge = gauge!(self.key, self.labels);
        gauge.increment(value);
    }
    pub fn decrement_by<T: IntoF64>(self, value: T) {
        let gauge = gauge!(self.key, self.labels);
        gauge.decrement(value);
    }
}

pub struct Empty {}
impl From<Empty> for Vec<Label> {
    fn from(_: Empty) -> Self {
        vec![]
    }
}
pub static OPERATION_GAUGE: Gauge<Empty> = Gauge {
    key: "operation_gauge",
    phantom: std::marker::PhantomData,
};
pub static EMBEDDING_REQUEST_GAUDGE: Gauge<Empty> = Gauge {
    key: "embedding_request_gauge",
    phantom: std::marker::PhantomData,
};
pub static PENDING_EMBEDDING_REQUEST_GAUDGE: Gauge<Empty> = Gauge {
    key: "pending_embedding_request_gauge",
    phantom: std::marker::PhantomData,
};
pub static JAVASCRIPT_REQUEST_GAUDGE: Gauge<Empty> = Gauge {
    key: "javascript_request_gauge",
    phantom: std::marker::PhantomData,
};
