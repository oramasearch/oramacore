#![allow(dead_code)]
use super::metrics;
use super::simd_metrics;
use core::{hash::Hash, iter::Sum};
use num::traits::{FromPrimitive, NumAssign};
use serde::{Deserialize, Serialize};

/// FloatElement trait, the generic of two primitive type `f32` and `f64`
///
pub trait FloatElement:
    FromPrimitive
    + Sized
    + Default
    + num::Zero
    + num::traits::FloatConst
    + core::fmt::Debug
    + Clone
    + Copy
    + PartialEq
    + PartialOrd
    + NumAssign
    + num::Signed
    + num::Float
    + Sync
    + Send
    + Sum
    + Serialize
    + simd_metrics::SIMDOptmized
{
    fn float_one() -> Self;

    fn float_two() -> Self;

    fn float_zero() -> Self;

    fn zero_patch_num() -> Self;
}

/// IdxType trait indicate the primitive type used for the data index
///
pub trait IdxType:
    Sized + Clone + Default + core::fmt::Debug + Eq + Ord + Sync + Send + Serialize + Hash
{
}

#[macro_export]
macro_rules! to_float_element {
    (  $x:ident  ) => {
        impl FloatElement for $x {
            fn float_one() -> Self {
                1.0
            }

            fn float_two() -> Self {
                1.0
            }

            fn float_zero() -> Self {
                0.0
            }

            fn zero_patch_num() -> Self {
                1.34e-6
            }
        }
    };
}

#[macro_export]
macro_rules! to_idx_type {
    (  $x:ident  ) => {
        impl IdxType for $x {}
    };
}

to_float_element!(f64);
to_float_element!(f32);
to_idx_type!(String);
to_idx_type!(usize);
to_idx_type!(i16);
to_idx_type!(i32);
to_idx_type!(i64);
to_idx_type!(i128);
to_idx_type!(u16);
to_idx_type!(u32);
to_idx_type!(u64);
to_idx_type!(u128);

/// Node is the main container for the point in the space
///
/// it contains a array of `FloatElement` and a index
///
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Node<E: FloatElement, T: IdxType> {
    vectors: Vec<E>,
    idx: Option<T>, // data id, it can be any type;
}

impl<E: FloatElement, T: IdxType> Node<E, T> {
    /// new without idx
    ///
    /// new a point without a idx
    pub fn new(vectors: &[E]) -> Node<E, T> {
        Node::<E, T>::valid_elements(vectors);
        Node {
            vectors: vectors.to_vec(),
            idx: Option::None,
        }
    }

    /// new with idx
    ///
    /// new a point with a idx
    pub fn new_with_idx(vectors: &[E], id: T) -> Node<E, T> {
        let mut n = Node::new(vectors);
        n.set_idx(id);
        n
    }

    /// calculate the point distance
    pub fn metric(&self, other: &Node<E, T>, t: metrics::Metric) -> Result<E, &'static str> {
        metrics::metric(&self.vectors, &other.vectors, t)
    }

    pub fn into_data(self) -> Option<(Vec<E>, T)> {
        self.idx.map(|id| (self.vectors, id))
    }

    // return internal embeddings
    pub fn vectors(&self) -> &Vec<E> {
        &self.vectors
    }

    // return mut internal embeddings
    pub fn mut_vectors(&mut self) -> &mut Vec<E> {
        &mut self.vectors
    }

    // set internal embeddings
    pub fn set_vectors(&mut self, v: &[E]) {
        self.vectors = v.to_vec();
    }

    // internal embeddings length
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    // return node's idx
    pub fn idx(&self) -> &Option<T> {
        &self.idx
    }

    fn set_idx(&mut self, id: T) {
        self.idx = Option::Some(id);
    }

    fn valid_elements(vectors: &[E]) -> bool {
        for e in vectors.iter() {
            if e.is_nan() || e.is_infinite() {
                //TODO: log
                panic!("invalid float element");
            }
        }
        true
    }
}

impl<E: FloatElement, T: IdxType> core::fmt::Display for Node<E, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "(key: {:#?}, vectors: {:#?})", self.idx, self.vectors)
    }
}

// general method

#[cfg(test)]
#[test]
fn node_test() {
    // f64
    let v = vec![1.0, 1.0];
    let v2 = vec![2.0, 2.0];
    let n = Node::<f64, usize>::new(&v);
    let n2 = Node::<f64, usize>::new(&v2);
    assert_eq!(n.metric(&n2, metrics::Metric::Manhattan).unwrap(), 2.0);
}
