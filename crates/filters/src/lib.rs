use fastbloom_rs::{BloomFilter, FilterBuilder, Membership};

pub trait DocId {
    fn as_u64(&self) -> u64;
}

pub struct PlainFilterResult<Id> {
    filter: BloomFilter,
    phantom: std::marker::PhantomData<Id>,
}

impl<Id: DocId> PlainFilterResult<Id> {
    pub fn new(expected_items: u64) -> Self {
        let filter = FilterBuilder::new(expected_items, 0.01).build_bloom_filter();

        PlainFilterResult {
            filter,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn ones(expected_items: u64) -> Self {
        let filter = FilterBuilder::new(expected_items, 0.01).build_bloom_filter();

        PlainFilterResult {
            filter,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn from_iter<I: Iterator<Item = Id>>(expected_items: u64, iter: I) -> Self {
        let filter = FilterBuilder::new(expected_items, 0.01).build_bloom_filter();

        let mut s = Self {
            filter,
            phantom: std::marker::PhantomData,
        };
        for id in iter {
            s.add(&id);
        }

        s
    }

    pub fn add(&mut self, id: &Id) {
        self.filter.add(&id.as_u64().to_be_bytes());
    }

    pub fn contains(&self, id: &Id) -> bool {
        self.filter.contains(&id.as_u64().to_be_bytes())
    }

    pub fn and(mut self, other: &Self) -> Self {
        self.filter.intersect(&other.filter);
        PlainFilterResult {
            filter: self.filter,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn or(mut self, other: &Self) -> Self {
        self.filter.union(&other.filter);
        PlainFilterResult {
            filter: self.filter,
            phantom: std::marker::PhantomData,
        }
    }
}

pub enum FilterResult<Id> {
    Filter(PlainFilterResult<Id>),
    And(Box<FilterResult<Id>>, Box<FilterResult<Id>>),
    Or(Box<FilterResult<Id>>, Box<FilterResult<Id>>),
    Not(Box<FilterResult<Id>>),
}

impl<Id: DocId> FilterResult<Id> {
    pub fn and(filter_1: FilterResult<Id>, filter_2: FilterResult<Id>) -> Self {
        match (filter_1, filter_2) {
            (FilterResult::Filter(filter1), FilterResult::Filter(filter2)) => {
                let output = filter1.and(&filter2);
                FilterResult::Filter(output)
            }
            (f1, f2) => FilterResult::And(Box::new(f1), Box::new(f2)),
        }
    }

    pub fn or(filter_1: FilterResult<Id>, filter_2: FilterResult<Id>) -> Self {
        match (filter_1, filter_2) {
            (FilterResult::Filter(filter1), FilterResult::Filter(filter2)) => {
                let output = filter1.or(&filter2);
                FilterResult::Filter(output)
            }
            (f1, f2) => FilterResult::Or(Box::new(f1), Box::new(f2)),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(filter: FilterResult<Id>) -> Self {
        FilterResult::Not(Box::new(filter))
    }

    pub fn contains(&self, id: &Id) -> bool {
        match self {
            FilterResult::Filter(filter) => filter.contains(id),
            FilterResult::And(filter_1, filter_2) => filter_1.contains(id) && filter_2.contains(id),
            FilterResult::Or(filter_1, filter_2) => filter_1.contains(id) || filter_2.contains(id),
            FilterResult::Not(filter) => !filter.contains(id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl DocId for u64 {
        #[inline]
        fn as_u64(&self) -> u64 {
            *self
        }
    }

    #[test]
    fn test_contains() {
        let mut filter = PlainFilterResult::new(1_000_000);
        for i in 0..100 {
            filter.add(&i);
        }

        for i in 0..100 {
            assert!(filter.contains(&i), "Expected {} to be true", i);
        }
        for i in 100..200 {
            assert!(!filter.contains(&i), "Expected {} to be false", i);
        }
    }

    fn build_bloom_filter(ids: Vec<u64>) -> PlainFilterResult<u64> {
        let mut filter = PlainFilterResult::new(1_000_000);
        for i in ids {
            filter.add(&i);
        }
        filter
    }

    #[test]
    fn test_plan_filter_simple() {
        let plan = FilterResult::Filter(build_bloom_filter(vec![2, 3, 4, 5]));

        assert!(!plan.contains(&1));
        assert!(plan.contains(&2));
        assert!(plan.contains(&3));
        assert!(plan.contains(&4));
        assert!(plan.contains(&5));
        assert!(!plan.contains(&6));
    }

    #[test]
    fn test_plan_filter_and() {
        let plan = FilterResult::And(
            Box::new(FilterResult::Filter(build_bloom_filter(vec![2, 3, 4, 5]))),
            Box::new(FilterResult::Filter(build_bloom_filter(vec![3, 4, 5, 6]))),
        );

        assert!(!plan.contains(&1));
        assert!(!plan.contains(&2));
        assert!(plan.contains(&3));
        assert!(plan.contains(&4));
        assert!(plan.contains(&5));
        assert!(!plan.contains(&6));
    }

    #[test]
    fn test_plan_filter_or() {
        let plan = FilterResult::Or(
            Box::new(FilterResult::Filter(build_bloom_filter(vec![2, 3, 4, 5]))),
            Box::new(FilterResult::Filter(build_bloom_filter(vec![3, 4, 5, 6]))),
        );

        assert!(!plan.contains(&1));
        assert!(plan.contains(&2));
        assert!(plan.contains(&3));
        assert!(plan.contains(&4));
        assert!(plan.contains(&5));
        assert!(plan.contains(&6));
        assert!(!plan.contains(&7));
    }

    #[test]
    fn test_plan_filter_and_not() {
        let plan = FilterResult::And(
            Box::new(FilterResult::Filter(build_bloom_filter(vec![2, 3, 4, 5]))),
            Box::new(FilterResult::Not(Box::new(FilterResult::Filter(
                build_bloom_filter(vec![3, 4, 5, 6]),
            )))),
        );

        assert!(!plan.contains(&1));
        assert!(plan.contains(&2));
        assert!(!plan.contains(&3));
        assert!(!plan.contains(&4));
        assert!(!plan.contains(&5));
        assert!(!plan.contains(&6));
        assert!(!plan.contains(&7));
    }

    #[test]
    fn test_plan_filter_or_not() {
        let plan = FilterResult::Or(
            Box::new(FilterResult::Filter(build_bloom_filter(vec![2, 3, 4, 5]))),
            Box::new(FilterResult::Not(Box::new(FilterResult::Filter(
                build_bloom_filter(vec![3, 4, 5, 6]),
            )))),
        );

        assert!(plan.contains(&1));
        assert!(plan.contains(&2));
        assert!(plan.contains(&3));
        assert!(plan.contains(&4));
        assert!(plan.contains(&5));
        assert!(!plan.contains(&6));
        assert!(plan.contains(&7));
    }
}
