use std::{cmp::Reverse, collections::BinaryHeap, fmt::Debug};

struct Item<K, V> {
    key: K,
    value: V,
}
impl<K: std::cmp::Ord, V: std::cmp::Ord> std::cmp::PartialEq for Item<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.value == other.value
    }
}
impl<K: std::cmp::Ord, V: std::cmp::Ord> std::cmp::Eq for Item<K, V> {}
impl<K: std::cmp::Ord, V: std::cmp::Ord> std::cmp::PartialOrd for Item<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: std::cmp::Ord, V: std::cmp::Ord> std::cmp::Ord for Item<K, V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| self.value.cmp(&other.value).reverse())
    }
}

impl<K: Debug, V: Debug> Debug for Item<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Item {{ key: {:?}, value: {:?} }}", self.key, self.value)
    }
}

pub struct CappedHeap<K, V> {
    heap: BinaryHeap<Reverse<Item<K, V>>>,
    limit: usize,
}

impl<K: std::cmp::Ord, V: std::cmp::Ord> CappedHeap<K, V> {
    pub fn new(limit: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            limit,
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        let new_item = Reverse(Item { key, value });
        if self.heap.len() < self.limit {
            self.heap.push(new_item);
        } else if let Some(i) = self.heap.peek() {
            if new_item < *i {
                self.heap.pop();
                self.heap.push(new_item);
            }
        }
    }

    pub fn into_top(self) -> impl Iterator<Item = (K, V)> {
        let v = self.heap.into_sorted_vec();
        v.into_iter()
            .map(|Reverse(Item { key, value })| (key, value))
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::SliceRandom;
    use rand::rng;

    use super::*;

    #[test]
    fn test_cappend_heap_item_order() {
        // Different key
        {
            let a = Item { key: 1, value: 1 };
            let b = Item { key: 2, value: 1 };

            assert!(b > a);
            assert!(a < b);

            let c = Item { key: 0, value: 1 };

            assert!(a > c);
            assert!(c < a);
            assert!(b > c);
            assert!(c < b);
        }

        // Same key different value
        {
            let a = Item { key: 0, value: 1 };
            let b = Item { key: 0, value: 2 };

            assert!(b < a);
            assert!(a > b);
        }
    }

    #[test]
    fn test_cappend_heap_foo() {
        let mut heap = CappedHeap::new(3);

        heap.insert(1, 1);
        heap.insert(2, 2);
        heap.insert(3, 3);
        heap.insert(4, 4);

        let top: Vec<_> = heap.into_top().collect();

        assert_eq!(top.len(), 3);
        assert_eq!(top, vec![(4, 4), (3, 3), (2, 2)]);
    }

    #[test]
    fn test_cappend_heap_order_consistency() {
        let mut data = vec![(1, 1), (1, 2), (1, 3), (1, 4)];

        for _ in 0..100 {
            let mut heap = CappedHeap::new(3);

            data.shuffle(&mut rng());
            for (key, value) in &data {
                heap.insert(*key, *value);
            }

            let top: Vec<_> = heap.into_top().collect();

            assert_eq!(top.len(), 3);
            assert_eq!(top, vec![(1, 1), (1, 2), (1, 3)]);
        }
    }
}
