use metrics::{gauge, IntoF64, Label};

#[macro_export]
macro_rules! create_gauge {
    ($name:ident, $key:expr, $type: ident) => {
        pub static $name: Gauge<$type> = Gauge {
            key: $key,
            phantom: std::marker::PhantomData,
        };
    };
}

pub struct Gauge<Labels> {
    pub key: &'static str,
    pub phantom: std::marker::PhantomData<Labels>,
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
