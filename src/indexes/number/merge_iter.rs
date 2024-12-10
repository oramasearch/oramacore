use std::{borrow::Cow, mem::replace};

#[derive(Copy, Clone, Debug)]
pub enum IterPointer {
    Iter1,
    Iter2,
}
impl IterPointer {
    fn reverse(&self) -> Self {
        match self {
            IterPointer::Iter1 => IterPointer::Iter2,
            IterPointer::Iter2 => IterPointer::Iter1,
        }
    }
}

pub enum MergeIterState<'a, K, V: Clone> {
    Processing,
    Unstarted,
    Running {
        current: IterPointer,
        cycle_on: (K, Cow<'a, Vec<V>>),
        cycle_on_index: usize,
        other_side: Option<(K, Cow<'a, Vec<V>>)>,
    },
    FetchFrom {
        fetch_from: IterPointer,
        other_side: Option<(K, Cow<'a, Vec<V>>)>,
    },
    Ended,
}

pub struct MergeIter<'a, K, V, I1, I2>
where
    I1: Iterator<Item = (K, Cow<'a, Vec<V>>)>,
    I2: Iterator<Item = (K, Cow<'a, Vec<V>>)>,
    K: Ord + Clone,
    V: Clone,
{
    pub iter1: I1,
    pub iter2: I2,
    pub state: MergeIterState<'a, K, V>,
}

impl<'a, K, V, I1, I2> Iterator for MergeIter<'a, K, V, I1, I2>
where
    I1: Iterator<Item = (K, Cow<'a, Vec<V>>)>,
    I2: Iterator<Item = (K, Cow<'a, Vec<V>>)>,
    K: Ord + Clone,
    V: Clone,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        let state = replace(&mut self.state, MergeIterState::Processing);
        match state {
            MergeIterState::Processing => unreachable!("Processing state should not be seen"),
            MergeIterState::Unstarted => {
                let other_side = self.iter2.next();
                self.state = MergeIterState::FetchFrom {
                    fetch_from: IterPointer::Iter1,
                    other_side,
                };
                return self.next();
            }
            MergeIterState::Running {
                current,
                cycle_on,
                cycle_on_index,
                other_side,
            } => {
                let k = cycle_on.0.clone();
                let v = cycle_on.1[cycle_on_index].clone();

                if cycle_on_index >= cycle_on.1.len() - 1 {
                    self.state = MergeIterState::FetchFrom {
                        fetch_from: current,
                        other_side,
                    };
                } else {
                    self.state = MergeIterState::Running {
                        current,
                        cycle_on,
                        cycle_on_index: cycle_on_index + 1,
                        other_side,
                    };
                }

                return Some((k, v));
            }
            MergeIterState::FetchFrom {
                fetch_from,
                other_side,
            } => {
                let iter = match fetch_from {
                    IterPointer::Iter1 => self.iter1.next(),
                    IterPointer::Iter2 => self.iter2.next(),
                };

                match (iter, other_side) {
                    (Some((iter_key, iter)), Some((other_side_key, other_size)))
                        if iter_key < other_side_key =>
                    {
                        self.state = MergeIterState::Running {
                            current: fetch_from,
                            cycle_on: (iter_key, iter),
                            cycle_on_index: 0,
                            other_side: Some((other_side_key, other_size)),
                        }
                    }
                    (Some((iter_key, iter)), Some((other_side_key, other_size)))
                        if iter_key >= other_side_key =>
                    {
                        self.state = MergeIterState::Running {
                            current: fetch_from.reverse(),
                            cycle_on: (other_side_key, other_size),
                            cycle_on_index: 0,
                            other_side: Some((iter_key, iter)),
                        }
                    }
                    (Some((iter_key, iter)), None) => {
                        self.state = MergeIterState::Running {
                            current: fetch_from,
                            cycle_on: (iter_key, iter),
                            cycle_on_index: 0,
                            other_side: None,
                        }
                    }
                    (None, Some((other_side_key, other_size))) => {
                        self.state = MergeIterState::Running {
                            current: fetch_from.reverse(),
                            cycle_on: (other_side_key, other_size),
                            cycle_on_index: 0,
                            other_side: None,
                        }
                    }
                    (Some(_), Some(_)) => {
                        panic!("This should not happen");
                    }
                    (None, None) => {
                        self.state = MergeIterState::Ended;
                    }
                };

                return self.next();
            }
            MergeIterState::Ended => {}
        };

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foo_bar_0() {
        let m = MergeIter {
            iter1: vec![
                (0, Cow::Owned(vec![1, 2, 3])),
                (1, Cow::Owned(vec![1, 2, 3])),
            ]
            .into_iter(),
            iter2: vec![
                (0, Cow::Owned(vec![1, 2, 3])),
                (2, Cow::Owned(vec![1, 2, 3])),
            ]
            .into_iter(),
            state: MergeIterState::Unstarted,
        };

        let v: Vec<_> = m.collect();

        assert_eq!(
            vec![
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 1),
                (2, 2),
                (2, 3),
            ],
            v
        );
    }

    #[test]
    fn test_foo_bar_4() {
        let m = MergeIter {
            iter1: vec![
                (1, Cow::Owned(vec![1, 2, 3])),
                (2, Cow::Owned(vec![1, 2, 3])),
            ]
            .into_iter(),
            iter2: vec![
                (0, Cow::Owned(vec![1, 2, 3])),
                (3, Cow::Owned(vec![1, 2, 3])),
            ]
            .into_iter(),
            state: MergeIterState::Unstarted,
        };

        let v: Vec<_> = m.collect();

        assert_eq!(
            vec![
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 1),
                (3, 2),
                (3, 3),
            ],
            v
        );
    }

    #[test]
    fn test_foo_bar_3() {
        let m = MergeIter {
            iter1: vec![
                (0, Cow::Owned(vec![1, 2, 3])),
                (3, Cow::Owned(vec![1, 2, 3])),
            ]
            .into_iter(),
            iter2: vec![
                (1, Cow::Owned(vec![1, 2, 3])),
                (2, Cow::Owned(vec![1, 2, 3])),
            ]
            .into_iter(),
            state: MergeIterState::Unstarted,
        };

        let v: Vec<_> = m.collect();

        assert_eq!(
            vec![
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 1),
                (3, 2),
                (3, 3),
            ],
            v
        );
    }

    #[test]
    fn test_foo_bar_2() {
        let m = MergeIter {
            iter1: vec![
                (0, Cow::Owned(vec![1, 2, 3])),
                (1, Cow::Owned(vec![1, 2, 3])),
            ]
            .into_iter(),
            iter2: vec![
                (2, Cow::Owned(vec![1, 2, 3])),
                (3, Cow::Owned(vec![1, 2, 3])),
            ]
            .into_iter(),
            state: MergeIterState::Unstarted,
        };

        let v: Vec<_> = m.collect();

        assert_eq!(
            vec![
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 1),
                (3, 2),
                (3, 3),
            ],
            v
        );
    }

    #[test]
    fn test_foo_bar_1() {
        let m = MergeIter {
            iter1: vec![
                (0, Cow::Owned(vec![1, 2, 3])),
                (2, Cow::Owned(vec![1, 2, 3])),
            ]
            .into_iter(),
            iter2: vec![
                (1, Cow::Owned(vec![1, 2, 3])),
                (3, Cow::Owned(vec![1, 2, 3])),
            ]
            .into_iter(),
            state: MergeIterState::Unstarted,
        };

        let v: Vec<_> = m.collect();

        assert_eq!(
            vec![
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 1),
                (3, 2),
                (3, 3),
            ],
            v
        );
    }
}
