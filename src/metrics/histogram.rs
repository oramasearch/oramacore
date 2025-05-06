use std::time::{Duration, Instant};

use metrics::{histogram, Label};
use num_traits::cast;
use tracing::error;

#[macro_export]
macro_rules! create_counter_histogram {
    ($name:ident, $key:expr, $type: ident) => {
        pub static $name: $crate::metrics::histogram::CounterHistogram<$type> =
            $crate::metrics::histogram::CounterHistogram {
                key: $key,
                phantom: std::marker::PhantomData,
            };
    };
}
#[macro_export]
macro_rules! create_time_histogram {
    ($name:ident, $key:expr, $type: ident) => {
        #[allow(dead_code)]
        pub static $name: $crate::metrics::histogram::TimeHistogram<$type> =
            $crate::metrics::histogram::TimeHistogram {
                key: $key,
                phantom: std::marker::PhantomData,
            };
    };
}

pub struct CounterHistogram<Labels> {
    pub key: &'static str,
    pub phantom: std::marker::PhantomData<Labels>,
}
impl<Labels: Into<Vec<Label>>> CounterHistogram<Labels> {
    pub fn track(&self, labels: Labels, value: f64) {
        histogram!(self.key, labels.into()).record(value);
    }
    pub fn track_usize(&self, labels: Labels, value: usize) {
        if let Some(value) = cast::<usize, f64>(value) {
            histogram!(self.key, labels.into()).record(value);
        } else {
            error!("Failed to cast value to f64");
        }
    }
}

pub struct TimeHistogram<Labels> {
    pub key: &'static str,
    pub phantom: std::marker::PhantomData<Labels>,
}
impl<Labels: Into<Vec<Label>>> TimeHistogram<Labels> {
    pub fn create(&self, labels: Labels) -> TimeHistogramImpl {
        TimeHistogramImpl {
            key: self.key,
            labels: labels.into(),
            created_at: Instant::now(),
            already_recorded: false,
        }
    }

    #[allow(dead_code)]
    pub fn track(&self, labels: Labels, duration: Duration) {
        let elapsed = duration.as_secs_f64();
        histogram!(self.key, labels.into()).record(elapsed);
    }
}
pub struct TimeHistogramImpl {
    key: &'static str,
    labels: Vec<Label>,
    created_at: Instant,
    already_recorded: bool,
}
impl Drop for TimeHistogramImpl {
    fn drop(&mut self) {
        if self.already_recorded {
            return;
        }
        self.already_recorded = true;
        let elapsed = self.created_at.elapsed().as_secs_f64();
        histogram!(self.key, self.labels.clone()).record(elapsed);
    }
}
