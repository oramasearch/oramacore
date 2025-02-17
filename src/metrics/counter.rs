use metrics::{counter, Label};
use num_traits::cast;
use tracing::error;

#[macro_export]
macro_rules! create_counter {
    ($name:ident, $key:expr, $type: ident) => {
        pub static $name: Counter<$type> = Counter {
            key: $key,
            phantom: std::marker::PhantomData,
        };
    };
}

pub struct Counter<Labels> {
    pub key: &'static str,
    pub phantom: std::marker::PhantomData<Labels>,
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