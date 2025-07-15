use std::iter::Peekable;

use crate::types::{DocumentId, SortOrder};

pub struct SortIterator<'s, T: Ord + Clone> {
    iter1: Peekable<Box<dyn Iterator<Item = (T, DocumentId)> + 's>>,
    iter2: Peekable<Box<dyn Iterator<Item = (T, DocumentId)> + 's>>,
    order: SortOrder,
}

impl<'s, T: Ord + Clone> SortIterator<'s, T> {
    pub fn new(
        iter1: Box<dyn Iterator<Item = (T, DocumentId)> + 's>,
        iter2: Box<dyn Iterator<Item = (T, DocumentId)> + 's>,
        order: SortOrder,
    ) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
            order,
        }
    }
}

impl<T: Ord + Clone> Iterator for SortIterator<'_, T> {
    type Item = DocumentId;

    fn next(&mut self) -> Option<Self::Item> {
        let el1 = self.iter1.peek();
        let el2 = self.iter2.peek();

        match (el1, el2) {
            (None, None) => None,
            (Some((_, _)), None) => self.iter1.next().map(|(_, v2)| v2),
            (None, Some((_, _))) => self.iter2.next().map(|(_, v1)| v1),
            (Some((k1, _)), Some((k2, _))) => match self.order {
                SortOrder::Ascending => {
                    if k1 < k2 {
                        self.iter1.next().map(|(_, v)| v)
                    } else {
                        self.iter2.next().map(|(_, v)| v)
                    }
                }
                SortOrder::Descending => {
                    if k1 > k2 {
                        self.iter1.next().map(|(_, v)| v)
                    } else {
                        self.iter2.next().map(|(_, v)| v)
                    }
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DocumentId, SortOrder};

    #[test]
    fn test_sort_iterator_ascending() {
        let data1 = vec![(1, 10), (3, 30), (5, 50)];
        let data2 = vec![(2, 20), (4, 40), (6, 60)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Ascending);
        let result: Vec<u64> = iter.map(|doc_id| doc_id.0).collect();
        assert_eq!(result, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn test_sort_iterator_descending() {
        let data1 = vec![(6, 60), (4, 40), (2, 20)];
        let data2 = vec![(5, 50), (3, 30), (1, 10)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Descending);
        let result: Vec<u64> = iter.map(|doc_id| doc_id.0).collect();
        assert_eq!(result, vec![60, 50, 40, 30, 20, 10]);
    }

    #[test]
    fn test_sort_iterator_mixed_lengths() {
        let data1 = vec![(1, 10), (3, 30)];
        let data2 = vec![(2, 20), (4, 40), (5, 50)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Ascending);
        let result: Vec<u64> = iter.map(|doc_id| doc_id.0).collect();
        assert_eq!(result, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_sort_iterator_with_duplicates() {
        // Both iterators contain the same key and value
        let data1 = vec![(1, 10), (2, 20), (3, 30)];
        let data2 = vec![(2, 20), (3, 30), (4, 40)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Ascending);
        let result: Vec<u64> = iter.map(|doc_id| doc_id.0).collect();

        assert_eq!(result, vec![10, 20, 20, 30, 30, 40]);
    }

    fn make_iter(data: Vec<(u64, u64)>) -> Box<dyn Iterator<Item = (u64, DocumentId)>> {
        Box::new(data.into_iter().map(|(k, v)| (k, DocumentId(v))))
    }
}
