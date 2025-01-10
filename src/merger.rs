use std::iter::Peekable;


pub struct MergedIterator<
    K,
    V1,
    V2,
    I1: Iterator<Item = (K, V1)>,
    I2: Iterator<Item = (K, V2)>,
    Transformer: FnMut(&K, V1) -> V2,
    Merger: FnMut(&K, V1, V2) -> V2,
> {
    iter1: Peekable<I1>,
    iter2: Peekable<I2>,
    transformer: Transformer,
    merger: Merger,
}

impl<
        K: Ord + Eq,
        V1,
        V2,
        I1: Iterator<Item = (K, V1)>,
        I2: Iterator<Item = (K, V2)>,
        Transformer: FnMut(&K, V1) -> V2,
        Merger: FnMut(&K, V1, V2) -> V2,
    > MergedIterator<K, V1, V2, I1, I2, Transformer, Merger>
{
    pub fn new(iter1: I1, iter2: I2, transformer: Transformer, merger: Merger) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
            transformer,
            merger,
        }
    }
}

impl<
        K: Ord + Eq,
        V1,
        V2,
        I1: Iterator<Item = (K, V1)>,
        I2: Iterator<Item = (K, V2)>,
        Transformer: FnMut(&K, V1) -> V2,
        Merger: FnMut(&K, V1, V2) -> V2,
    > Iterator for MergedIterator<K, V1, V2, I1, I2, Transformer, Merger>
{
    type Item = (K, V2);

    fn next(&mut self) -> Option<Self::Item> {
        let first = self.iter1.peek();
        let second = self.iter2.peek();

        match (first, second) {
            (Some((k1, _)), Some((k2, _))) => {
                let cmp = k1.cmp(k2);
                match cmp {
                    std::cmp::Ordering::Less => {
                        let v = self.iter1.next();
                        if let Some((k, v)) = v {
                            let v = (self.transformer)(&k, v);
                            Some((k, v))
                        } else {
                            None
                        }
                    }
                    std::cmp::Ordering::Greater => self.iter2.next(),
                    std::cmp::Ordering::Equal => {
                        let (k1, v1) = self.iter1.next().unwrap();
                        let (_, v2) = self.iter2.next().unwrap();
                        let v = (self.merger)(&k1, v1, v2);
                        Some((k1, v))
                    }
                }
            }
            (Some(_), None) => {
                let v = self.iter1.next();
                if let Some((k, v)) = v {
                    let v = (self.transformer)(&k, v);
                    Some((k, v))
                } else {
                    None
                }
            }
            (None, Some(_)) => self.iter2.next(),
            (None, None) => None,
        }
    }
}
