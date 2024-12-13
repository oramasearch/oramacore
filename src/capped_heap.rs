use std::{cmp::Reverse, collections::BinaryHeap};

pub struct CappedHeap<K, V> {
    heap: BinaryHeap<Reverse<(K, V)>>,
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
        if self.heap.len() < self.limit {
            self.heap.push(Reverse((key, value)));
        } else if let Some(Reverse((min_key, _))) = self.heap.peek() {
            if key > *min_key {
                self.heap.pop();
                self.heap.push(Reverse((key, value)));
            }
        }
    }

    pub fn into_top(self) -> Vec<Reverse<(K, V)>> {
        self.heap.into_sorted_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cappend_heap() {
        let mut heap = CappedHeap::new(3);

        heap.insert(1, 1);
        heap.insert(2, 2);
        heap.insert(3, 3);
        heap.insert(4, 4);

        let top = heap.into_top();

        assert_eq!(top.len(), 3);
        assert_eq!(top, vec![Reverse((4, 4)), Reverse((3, 3)), Reverse((2, 2))]);
    }
}
