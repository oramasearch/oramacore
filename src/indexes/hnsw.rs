/**
 * This file contains a copy of the HNSW implementation from `instant-distance` crate.
 * It is also released under the MIT license.
 * See the original repo here:
 * https://github.com/djc/instant-distance/tree/84745917de862ca02da02105549c6e13edd8d3ac/instant-distance
 * Some part of this code comes also from here:
 * https://github.com/djc/instant-distance/pull/40/files
 */
use std::hash::Hash;
use std::mem;
use std::ops::{Deref, Index};

use ordered_float::OrderedFloat;
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};
use serde::{Deserialize, Serialize};

use std::cmp::{max, Ordering, Reverse};
use std::collections::BinaryHeap;
use std::collections::HashSet;

use parking_lot::Mutex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use serde_big_array::BigArray;

#[derive(Clone)]
/// Parameters for building the `Hnsw`
pub struct Builder {
    ef_search: usize,
    ef_construction: usize,
    heuristic: Option<Heuristic>,
    ml: f32,
    seed: u64,
}

impl Builder {
    /// Set the `efConstruction` parameter from the paper
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set the `ef` parameter from the paper
    ///
    /// If the `efConstruction` parameter is not already set, it will be set
    /// to the same value as `ef` by default.
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    pub fn select_heuristic(mut self, params: Option<Heuristic>) -> Self {
        self.heuristic = params;
        self
    }

    /// Set the `mL` parameter from the paper
    ///
    /// If the `mL` parameter is not already set, it defaults to `1.0 / ln(M)`.
    pub fn ml(mut self, ml: f32) -> Self {
        self.ml = ml;
        self
    }

    /// Set the seed value for the random number generator used to generate a layer for each point
    ///
    /// If this value is left unset, a seed is generated from entropy (via `getrandom()`).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Build an `HnswMap` with the given sets of points and values
    pub fn build<P: Point, V: Clone>(self, points: Vec<P>, values: Vec<V>) -> HnswMap<P, V> {
        HnswMap::new(points, values, self)
    }

    /// Build the `Hnsw` with the given set of points
    pub fn build_hnsw<P: Point>(self, points: Vec<P>) -> (Hnsw<P>, Vec<PointId>) {
        Hnsw::new(points, self)
    }

    pub fn into_parts(self) -> (usize, usize, f32, u64) {
        let Self {
            ef_search,
            ef_construction,
            heuristic: _,
            ml,
            seed,
            ..
        } = self;
        (ef_search, ef_construction, ml, seed)
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            ef_search: 100,
            ef_construction: 100,
            heuristic: Some(Heuristic::default()),
            ml: 1.0 / (M as f32).ln(),
            seed: rand::random(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Heuristic {
    pub extend_candidates: bool,
    pub keep_pruned: bool,
}

impl Default for Heuristic {
    fn default() -> Self {
        Heuristic {
            extend_candidates: false,
            keep_pruned: true,
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct HnswMap<P, V> {
    hnsw: Hnsw<P>,
    ef_construction: usize,
    values: Vec<V>,
}

impl<P, V> HnswMap<P, V>
where
    P: Point,
    V: Clone,
{
    fn new(points: Vec<P>, values: Vec<V>, builder: Builder) -> Self {
        let ef_construction = builder.ef_construction;
        let (hnsw, ids) = Hnsw::new(points, builder);

        let mut sorted = ids.into_iter().enumerate().collect::<Vec<_>>();
        sorted.sort_unstable_by(|a, b| a.1.cmp(&b.1));
        let new = sorted
            .into_iter()
            .map(|(src, _)| values[src].clone())
            .collect();

        Self {
            hnsw,
            values: new,
            ef_construction,
        }
    }

    pub fn len(&self) -> usize {
        self.hnsw.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn search<'a>(
        &'a self,
        point: &P,
        search: &'a mut Search,
    ) -> impl ExactSizeIterator<Item = MapItem<'a, P, V>> + 'a {
        self.hnsw
            .search(point, search)
            .map(move |item| MapItem::from(item, self))
    }

    /// Iterate over the keys and values in this index
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &P)> {
        self.hnsw.iter()
    }

    pub fn get(&self, i: usize, search: &Search) -> Option<MapItem<'_, P, V>> {
        Some(MapItem::from(self.hnsw.get(i, search)?, self))
    }

    pub fn insert_multiple(&mut self, points: Vec<P>, values: Vec<V>) {
        self.hnsw
            .insert_multiple(points, self.ef_construction, Some(Heuristic::default()));
        self.values.extend(values);
    }
}

fn hnsw_map_into_iter_map<P, V>(((p, _), v): ((P, PointId), V)) -> (P, V) {
    (p, v)
}

impl<P, V> IntoIterator for HnswMap<P, V> {
    type Item = (P, V);
    type IntoIter = std::iter::Map<
        std::iter::Zip<
            std::iter::Map<
                std::iter::Enumerate<std::vec::IntoIter<P>>,
                fn((usize, P)) -> (P, PointId),
            >,
            std::vec::IntoIter<V>,
        >,
        fn(((P, PointId), V)) -> (P, V),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.hnsw
            .into_iter()
            .zip(self.values)
            .map(hnsw_map_into_iter_map)
    }
}

pub struct MapItem<'a, P, V> {
    pub distance: f32,
    pub pid: PointId,
    pub point: &'a P,
    pub value: &'a V,
}

impl<'a, P, V> MapItem<'a, P, V> {
    fn from(item: Item<'a, P>, map: &'a HnswMap<P, V>) -> Self {
        MapItem {
            distance: item.distance,
            pid: item.pid,
            point: item.point,
            value: &map.values[item.pid.0 as usize],
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct Hnsw<P> {
    ef_search: usize,
    points: Vec<P>,
    zero: Vec<ZeroNode>,
    layers: Vec<Vec<UpperNode>>,
}

impl<P> Hnsw<P>
where
    P: Point,
{
    pub fn builder() -> Builder {
        Builder::default()
    }

    fn new(points: Vec<P>, builder: Builder) -> (Self, Vec<PointId>) {
        let ef_search = builder.ef_search;
        let ef_construction = builder.ef_construction;
        let ml = builder.ml;
        let heuristic = builder.heuristic;
        let mut rng = SmallRng::seed_from_u64(builder.seed);

        if points.is_empty() {
            return (
                Self {
                    ef_search,
                    zero: Vec::new(),
                    points: Vec::new(),
                    layers: Vec::new(),
                },
                Vec::new(),
            );
        }

        // Determine the number and size of layers.

        let mut sizes = Vec::new();
        let mut num = points.len();
        loop {
            let next = (num as f32 * ml) as usize;
            if next < M {
                break;
            }
            sizes.push((num - next, num));
            num = next;
        }
        sizes.push((num, num));
        sizes.reverse();
        let top = LayerId(sizes.len() - 1);

        // Give all points a random layer and sort the list of nodes by descending order for
        // construction. This allows us to copy higher layers to lower layers as construction
        // progresses, while preserving randomness in each point's layer and insertion order.

        assert!(points.len() < u32::MAX as usize);
        let mut shuffled = (0..points.len())
            .map(|i| (PointId(rng.random_range(0..points.len() as u32)), i))
            .collect::<Vec<_>>();
        shuffled.sort_unstable();

        let mut out = vec![INVALID; points.len()];
        let points = shuffled
            .into_iter()
            .enumerate()
            .map(|(i, (_, idx))| {
                out[idx] = PointId(i as u32);
                points[idx].clone()
            })
            .collect::<Vec<_>>();

        // Figure out how many nodes will go on each layer. This helps us allocate memory capacity
        // for each layer in advance, and also helps enable batch insertion of points.

        let num_layers = sizes.len();
        let mut ranges = Vec::with_capacity(top.0);
        for (i, (size, cumulative)) in sizes.into_iter().enumerate() {
            let start = cumulative - size;
            // Skip the first point, since we insert the enter point separately
            ranges.push((LayerId(num_layers - i - 1), max(start, 1)..cumulative));
        }

        // Initialize data for layers

        let mut layers = vec![vec![]; top.0];
        let zero = points
            .iter()
            .map(|_| RwLock::new(ZeroNode::default()))
            .collect::<Vec<_>>();

        let state = Construction {
            zero: zero.as_slice(),
            pool: SearchPool::new(points.len()),
            top,
            points: &points,
            heuristic,
            ef_construction,
        };

        for (layer, range) in ranges {
            let inserter = |pid| state.insert(pid, layer, &layers);

            let end = range.end;
            if layer == top {
                range.into_iter().for_each(|i| inserter(PointId(i as u32)))
            } else {
                range
                    .into_par_iter()
                    .for_each(|i| inserter(PointId(i as u32)));
            }

            // For layers above the zero layer, make a copy of the current state of the zero layer
            // with `nearest` truncated to `M` elements.
            if !layer.is_zero() {
                (&state.zero[..end])
                    .into_par_iter()
                    .map(|zero| UpperNode::from_zero(&zero.read()))
                    .collect_into_vec(&mut layers[layer.0 - 1]);
            }
        }

        (
            Self {
                ef_search,
                zero: zero.into_iter().map(|node| node.into_inner()).collect(),
                points,
                layers,
            },
            out,
        )
    }

    /// Search the index for the points nearest to the reference point `point`
    ///
    /// The results are returned in the `out` parameter; the number of neighbors to search for
    /// is limited by the size of the `out` parameter, and the number of results found is returned
    /// in the return value.
    pub fn search<'a, 'b: 'a>(
        &'b self,
        point: &P,
        search: &'a mut Search,
    ) -> impl ExactSizeIterator<Item = Item<'b, P>> + 'a {
        search.reset();
        let map = move |candidate| Item::new(candidate, self);
        if self.points.is_empty() {
            return search.iter().map(map);
        }

        search.visited.reserve_capacity(self.points.len());
        search.push(PointId(0), point, &self.points);
        for cur in LayerId(self.layers.len()).descend() {
            let (ef, num) = match cur.is_zero() {
                true => (self.ef_search, M * 2),
                false => (1, M),
            };

            search.ef = ef;
            match cur.0 {
                0 => search.search(point, self.zero.as_slice(), &self.points, num),
                l => search.search(point, self.layers[l - 1].as_slice(), &self.points, num),
            }

            if !cur.is_zero() {
                search.cull();
            }
        }

        search.iter().map(map)
    }

    /// Iterate over the keys and values in this index
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &P)> {
        self.points
            .iter()
            .enumerate()
            .map(|(i, p)| (PointId(i as u32), p))
    }

    pub fn insert_multiple(
        &mut self,
        points: Vec<P>,
        ef_construction: usize,
        heuristic: Option<Heuristic>,
    ) -> PointId {
        let new_pid = self.points.len();
        let new_point_id = PointId(new_pid as u32);

        self.points.extend(points);
        self.zero.push(ZeroNode::default());

        let taken = mem::take(&mut self.zero);
        let zeros = taken.into_iter().map(RwLock::new).collect::<Vec<_>>();

        let top = if self.layers.is_empty() {
            LayerId(0)
        } else {
            LayerId(self.layers.len())
        };

        let construction = Construction {
            zero: zeros.as_slice(),
            pool: SearchPool::new(self.points.len()),
            top,
            points: self.points.as_slice(),
            heuristic,
            ef_construction,
        };

        let new_layer = construction.top;
        construction.insert(new_point_id, new_layer, &self.layers);

        self.zero = construction.zero.iter().map(|node| *node.read()).collect();

        new_point_id
    }

    #[doc(hidden)]
    pub fn get(&self, i: usize, search: &Search) -> Option<Item<'_, P>> {
        Some(Item::new(search.nearest.get(i).copied()?, self))
    }
}

fn hnsw_into_iter_map<P>((i, p): (usize, P)) -> (P, PointId) {
    (p, PointId(i as u32))
}

impl<P> IntoIterator for Hnsw<P> {
    type Item = (P, PointId);
    type IntoIter =
        std::iter::Map<std::iter::Enumerate<std::vec::IntoIter<P>>, fn((usize, P)) -> (P, PointId)>;

    fn into_iter(self) -> Self::IntoIter {
        self.points.into_iter().enumerate().map(hnsw_into_iter_map)
    }
}

pub struct Item<'a, P> {
    pub distance: f32,
    pub pid: PointId,
    pub point: &'a P,
}

impl<'a, P> Item<'a, P> {
    fn new(candidate: Candidate, hnsw: &'a Hnsw<P>) -> Self {
        Self {
            distance: candidate.distance.into_inner(),
            pid: candidate.pid,
            point: &hnsw[candidate.pid],
        }
    }
}

struct Construction<'a, P: Point> {
    zero: &'a [RwLock<ZeroNode>],
    pool: SearchPool,
    top: LayerId,
    points: &'a [P],
    heuristic: Option<Heuristic>,
    ef_construction: usize,
}

impl<P: Point> Construction<'_, P> {
    /// Insert new node in the zero layer
    ///
    /// * `new` is the `PointId` for the new node
    /// * `layer` contains all the nodes at the current layer
    /// * `layers` refers to the existing higher-level layers
    ///
    /// Creates the new node, initializing its `nearest` array and updates the nearest neighbors
    /// for the new node's neighbors if necessary before appending the new node to the layer.
    fn insert(&self, new: PointId, layer: LayerId, layers: &[Vec<UpperNode>]) {
        let mut node = self.zero[new].write();
        let (mut search, mut insertion) = self.pool.pop();
        insertion.ef = self.ef_construction;

        let point = &self.points[new];
        search.reset();
        search.push(PointId(0), point, self.points);
        let num = if layer.is_zero() { M * 2 } else { M };

        for cur in self.top.descend() {
            search.ef = if cur <= layer {
                self.ef_construction
            } else {
                1
            };
            match cur > layer {
                true => {
                    search.search(point, layers[cur.0 - 1].as_slice(), self.points, num);
                    search.cull();
                }
                false => {
                    search.search(point, self.zero, self.points, num);
                    break;
                }
            }
        }

        let found = match self.heuristic {
            None => {
                let candidates = search.select_simple();
                &candidates[..Ord::min(candidates.len(), M * 2)]
            }
            Some(heuristic) => {
                search.select_heuristic(&self.points[new], self.zero, self.points, heuristic)
            }
        };

        // Just make sure the candidates are all unique
        debug_assert_eq!(
            found.len(),
            found.iter().map(|c| c.pid).collect::<HashSet<_>>().len()
        );

        for (i, candidate) in found.iter().enumerate() {
            // `candidate` here is the new node's neighbor
            let &Candidate { distance, pid } = candidate;
            if let Some(heuristic) = self.heuristic {
                let found = insertion.add_neighbor_heuristic(
                    new,
                    self.zero.nearest_iter(pid),
                    self.zero,
                    &self.points[pid],
                    self.points,
                    heuristic,
                );

                self.zero[pid]
                    .write()
                    .rewrite(found.iter().map(|candidate| candidate.pid));
            } else {
                // Find the correct index to insert at to keep the neighbor's neighbors sorted
                let old = &self.points[pid];
                let idx = self.zero[pid]
                    .read()
                    .binary_search_by(|third| {
                        // `third` here is one of the neighbors of the new node's neighbor.
                        let third = match third {
                            pid if pid.is_valid() => *pid,
                            // if `third` is `None`, our new `node` is always "closer"
                            _ => return Ordering::Greater,
                        };

                        distance.cmp(&old.distance(&self.points[third]).into())
                    })
                    .unwrap_or_else(|e| e);

                self.zero[pid].write().insert(idx, new);
            }
            node.set(i, pid);
        }

        self.pool.push((search, insertion));
    }
}

struct SearchPool {
    pool: Mutex<Vec<(Search, Search)>>,
    len: usize,
}

impl SearchPool {
    fn new(len: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            len,
        }
    }

    fn pop(&self) -> (Search, Search) {
        match self.pool.lock().pop() {
            Some(res) => res,
            None => (Search::new(self.len), Search::new(self.len)),
        }
    }

    fn push(&self, item: (Search, Search)) {
        self.pool.lock().push(item);
    }
}

/// Keeps mutable state for searching a point's nearest neighbors
///
/// In particular, this contains most of the state used in algorithm 2. The structure is
/// initialized by using `push()` to add the initial enter points.
pub struct Search {
    /// Nodes visited so far (`v` in the paper)
    visited: Visited,
    /// Candidates for further inspection (`C` in the paper)
    candidates: BinaryHeap<Reverse<Candidate>>,
    /// Nearest neighbors found so far (`W` in the paper)
    ///
    /// This must always be in sorted (nearest first) order.
    nearest: Vec<Candidate>,
    /// Working set for heuristic selection
    working: Vec<Candidate>,
    discarded: Vec<Candidate>,
    /// Maximum number of nearest neighbors to retain (`ef` in the paper)
    ef: usize,
}

impl Search {
    fn new(capacity: usize) -> Self {
        Self {
            visited: Visited::with_capacity(capacity),
            ..Default::default()
        }
    }

    /// Search the given layer for nodes near the given `point`
    ///
    /// This contains the loops from the paper's algorithm 2. `point` represents `q`, the query
    /// element; `search.candidates` contains the enter points `ep`. `points` contains all the
    /// points, which is required to calculate distances between two points.
    ///
    /// The `links` argument represents the number of links from each candidate to consider. This
    /// function may be called for a higher layer (with M links per node) or the zero layer (with
    /// M * 2 links per node), but for performance reasons we often call this function on the data
    /// representation matching the zero layer even when we're referring to a higher layer. In that
    /// case, we use `links` to constrain the number of per-candidate links we consider for search.
    ///
    /// Invariants: `self.nearest` should be in sorted (nearest first) order, and should be
    /// truncated to `self.ef`.
    fn search<L: Layer, P: Point>(&mut self, point: &P, layer: L, points: &[P], links: usize) {
        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if let Some(furthest) = self.nearest.last() {
                if candidate.distance > furthest.distance {
                    break;
                }
            }

            for pid in layer.nearest_iter(candidate.pid).take(links) {
                self.push(pid, point, points);
            }

            // If we don't truncate here, `furthest` will be further out than necessary, making
            // us continue looping while we could have broken out.
            self.nearest.truncate(self.ef);
        }
    }

    fn add_neighbor_heuristic<L: Layer, P: Point>(
        &mut self,
        new: PointId,
        current: impl Iterator<Item = PointId>,
        layer: L,
        point: &P,
        points: &[P],
        params: Heuristic,
    ) -> &[Candidate] {
        self.reset();
        self.push(new, point, points);
        for pid in current {
            self.push(pid, point, points);
        }
        self.select_heuristic(point, layer, points, params)
    }

    /// Heuristically sort and truncate neighbors in `self.nearest`
    ///
    /// Invariant: `self.nearest` must be in sorted (nearest first) order.
    fn select_heuristic<L: Layer, P: Point>(
        &mut self,
        point: &P,
        layer: L,
        points: &[P],
        params: Heuristic,
    ) -> &[Candidate] {
        self.working.clear();
        // Get input candidates from `self.nearest` and store them in `self.working`.
        // `self.candidates` will represent `W` from the paper's algorithm 4 for now.
        for &candidate in &self.nearest {
            self.working.push(candidate);
            if params.extend_candidates {
                for hop in layer.nearest_iter(candidate.pid) {
                    if !self.visited.insert(hop) {
                        continue;
                    }

                    let other = &points[hop];
                    let distance = OrderedFloat::from(point.distance(other));
                    let new = Candidate { distance, pid: hop };
                    self.working.push(new);
                }
            }
        }

        if params.extend_candidates {
            self.working.sort_unstable();
        }

        self.nearest.clear();
        self.discarded.clear();
        for candidate in self.working.drain(..) {
            if self.nearest.len() >= M * 2 {
                break;
            }

            // Disadvantage candidates which are closer to an existing result point than they
            // are to the query point, to facilitate bridging between clustered points.
            let candidate_point = &points[candidate.pid];
            let nearest = !self.nearest.iter().any(|result| {
                let distance = OrderedFloat::from(candidate_point.distance(&points[result.pid]));
                distance < candidate.distance
            });

            match nearest {
                true => self.nearest.push(candidate),
                false => self.discarded.push(candidate),
            }
        }

        if params.keep_pruned {
            // Add discarded connections from `working` (`Wd`) to `self.nearest` (`R`)
            for candidate in self.discarded.drain(..) {
                if self.nearest.len() >= M * 2 {
                    break;
                }
                self.nearest.push(candidate);
            }
        }

        &self.nearest
    }

    /// Track node `pid` as a potential new neighbor for the given `point`
    ///
    /// Will immediately return if the node has been considered before. This implements
    /// the inner loop from the paper's algorithm 2.
    fn push<P: Point>(&mut self, pid: PointId, point: &P, points: &[P]) {
        if !self.visited.insert(pid) {
            return;
        }

        let other = &points[pid];
        let distance = OrderedFloat::from(point.distance(other));
        let new = Candidate { distance, pid };
        let idx = match self.nearest.binary_search(&new) {
            Err(idx) if idx < self.ef => idx,
            Err(_) => return,
            Ok(_) => unreachable!(),
        };

        self.nearest.insert(idx, new);
        self.candidates.push(Reverse(new));
    }

    /// Lower the search to the next lower level
    ///
    /// Re-initialize the `Search`: `nearest`, the output `W` from the last round, now becomes
    /// the set of enter points, which we use to initialize both `candidates` and `visited`.
    ///
    /// Invariant: `nearest` should be sorted and truncated before this is called. This is generally
    /// the case because `Layer::search()` is always called right before calling `cull()`.
    fn cull(&mut self) {
        self.candidates.clear();
        for &candidate in self.nearest.iter() {
            self.candidates.push(Reverse(candidate));
        }

        self.visited.clear();
        self.visited.extend(self.nearest.iter().map(|c| c.pid));
    }

    /// Resets the state to be ready for a new search
    fn reset(&mut self) {
        let Search {
            visited,
            candidates,
            nearest,
            working,
            discarded,
            ef: _,
        } = self;

        visited.clear();
        candidates.clear();
        nearest.clear();
        working.clear();
        discarded.clear();
    }

    /// Selection of neighbors for insertion (algorithm 3 from the paper)
    fn select_simple(&mut self) -> &[Candidate] {
        &self.nearest
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = Candidate> + '_ {
        self.nearest.iter().copied()
    }
}

impl Default for Search {
    fn default() -> Self {
        Self {
            visited: Visited::with_capacity(0),
            candidates: BinaryHeap::new(),
            nearest: Vec::new(),
            working: Vec::new(),
            discarded: Vec::new(),
            ef: 1,
        }
    }
}

pub trait Point: Clone + Sync {
    fn distance(&self, other: &Self) -> f32;
}

/// The parameter `M` from the paper
///
/// This should become a generic argument to `Hnsw` when possible.
const M: usize = 32;

pub(crate) struct Visited {
    store: Vec<u8>,
    generation: u8,
}

impl Visited {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            store: vec![0; capacity],
            generation: 1,
        }
    }

    pub(crate) fn reserve_capacity(&mut self, capacity: usize) {
        if self.store.len() != capacity {
            self.store.resize(capacity, self.generation - 1);
        }
    }

    pub(crate) fn insert(&mut self, pid: PointId) -> bool {
        let slot = &mut self.store[pid.0 as usize];
        if *slot != self.generation {
            *slot = self.generation;
            true
        } else {
            false
        }
    }

    pub(crate) fn extend(&mut self, iter: impl Iterator<Item = PointId>) {
        for pid in iter {
            self.insert(pid);
        }
    }

    pub(crate) fn clear(&mut self) {
        if self.generation < 249 {
            self.generation += 1;
            return;
        }

        let len = self.store.len();
        self.store.clear();
        self.store.resize(len, 0);
        self.generation = 1;
    }
}

#[derive(Deserialize, Serialize, Clone, Copy, Debug, Default)]
pub(crate) struct UpperNode([PointId; M]);

impl UpperNode {
    pub(crate) fn from_zero(node: &ZeroNode) -> Self {
        let mut nearest = [INVALID; M];
        nearest.copy_from_slice(&node.0[..M]);
        Self(nearest)
    }
}

impl<'a> Layer for &'a [UpperNode] {
    type Slice = &'a [PointId];

    fn nearest_iter(&self, pid: PointId) -> NearestIter<Self::Slice> {
        NearestIter::new(&self[pid.0 as usize].0)
    }
}

#[derive(Deserialize, Serialize, Clone, Copy, Debug)]
pub(crate) struct ZeroNode(#[serde(with = "BigArray")] pub(crate) [PointId; M * 2]);

impl ZeroNode {
    pub(crate) fn rewrite(&mut self, mut iter: impl Iterator<Item = PointId>) {
        for slot in self.0.iter_mut() {
            if let Some(pid) = iter.next() {
                *slot = pid;
            } else if *slot != INVALID {
                *slot = INVALID;
            } else {
                break;
            }
        }
    }

    pub(crate) fn insert(&mut self, idx: usize, pid: PointId) {
        // It might be possible for all the neighbor's current neighbors to be closer to our
        // neighbor than to the new node, in which case we skip insertion of our new node's ID.
        if idx >= self.0.len() {
            return;
        }

        if self.0[idx].is_valid() {
            let end = (M * 2) - 1;
            self.0.copy_within(idx..end, idx + 1);
        }

        self.0[idx] = pid;
    }

    pub(crate) fn set(&mut self, idx: usize, pid: PointId) {
        self.0[idx] = pid;
    }
}

impl Default for ZeroNode {
    fn default() -> ZeroNode {
        ZeroNode([INVALID; M * 2])
    }
}

impl Deref for ZeroNode {
    type Target = [PointId];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> Layer for &'a [ZeroNode] {
    type Slice = &'a [PointId];

    fn nearest_iter(&self, pid: PointId) -> NearestIter<Self::Slice> {
        NearestIter::new(&self[pid.0 as usize])
    }
}

impl<'a> Layer for &'a [RwLock<ZeroNode>] {
    type Slice = MappedRwLockReadGuard<'a, [PointId]>;

    fn nearest_iter(&self, pid: PointId) -> NearestIter<Self::Slice> {
        NearestIter::new(RwLockReadGuard::map(
            self[pid.0 as usize].read(),
            Deref::deref,
        ))
    }
}

pub(crate) trait Layer {
    type Slice: Deref<Target = [PointId]>;
    fn nearest_iter(&self, pid: PointId) -> NearestIter<Self::Slice>;
}

pub(crate) struct NearestIter<T> {
    node: T,
    cur: usize,
}

impl<T> NearestIter<T>
where
    T: Deref<Target = [PointId]>,
{
    fn new(node: T) -> Self {
        Self { node, cur: 0 }
    }
}

impl<T> Iterator for NearestIter<T>
where
    T: Deref<Target = [PointId]>,
{
    type Item = PointId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.node.len() {
            return None;
        }

        let item = self.node[self.cur];
        if !item.is_valid() {
            self.cur = self.node.len();
            return None;
        }

        self.cur += 1;
        Some(item)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct LayerId(pub usize);

impl LayerId {
    pub(crate) fn descend(&self) -> impl Iterator<Item = LayerId> {
        DescendingLayerIter { next: Some(self.0) }
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

struct DescendingLayerIter {
    next: Option<usize>,
}

impl Iterator for DescendingLayerIter {
    type Item = LayerId;

    fn next(&mut self) -> Option<Self::Item> {
        Some(LayerId(match self.next? {
            0 => {
                self.next = None;
                0
            }
            next => {
                self.next = Some(next - 1);
                next
            }
        }))
    }
}

/// A potential nearest neighbor
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Candidate {
    pub(crate) distance: OrderedFloat<f32>,
    /// The identifier for the neighboring point
    pub pid: PointId,
}

/// References a `Point` in the `Hnsw`
///
/// This can be used to index into the `Hnsw` to refer to the `Point` data.
#[derive(Deserialize, Serialize, Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PointId(pub(crate) u32);

impl PointId {
    /// Whether this value represents a valid point
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }

    /// Return the identifier value
    pub fn into_inner(self) -> u32 {
        self.0
    }
}

// Not part of the public API; only for use in bindings
impl From<u32> for PointId {
    fn from(id: u32) -> Self {
        PointId(id)
    }
}

impl Default for PointId {
    fn default() -> Self {
        INVALID
    }
}

impl<P> Index<PointId> for Hnsw<P> {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self.points[index.0 as usize]
    }
}

impl<P: Point> Index<PointId> for [P] {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl Index<PointId> for [RwLock<ZeroNode>] {
    type Output = RwLock<ZeroNode>;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

pub(crate) const INVALID: PointId = PointId(u32::MAX);

#[cfg(test)]
mod tests {
    use rand::rngs::ThreadRng;

    use super::*;

    #[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct Point(isize, isize, isize);

    impl super::Point for Point {
        fn distance(&self, other: &Self) -> f32 {
            // Euclidean distance metric
            (((self.0 - other.0).pow(2) + (self.1 - other.1).pow(2) + (self.2 - other.2).pow(2))
                as f32)
                .sqrt()
        }
    }

    #[test]
    fn test_hnsw() {
        let points = vec![Point(255, 0, 0), Point(0, 255, 0), Point(0, 0, 255)];
        let values = vec!["red", "green", "blue"];

        let map: HnswMap<Point, &str> = Builder::default().build(points, values);
        let mut search = Search::default();
        let cambridge_blue = Point(163, 193, 173);
        let closest_point_1 = map.search(&cambridge_blue, &mut search).next().unwrap();

        let ser = serde_json::to_string(&map).unwrap();
        let map: HnswMap<Point, &str> = serde_json::from_str(&ser).unwrap();

        let mut search = Search::default();
        let cambridge_blue = Point(163, 193, 173);
        let closest_point_2 = map.search(&cambridge_blue, &mut search).next().unwrap();

        assert_eq!(closest_point_1.pid, closest_point_2.pid);
        assert_eq!(closest_point_1.point, closest_point_2.point);
        assert_eq!(closest_point_1.value, closest_point_2.value);
    }

    #[test]
    #[allow(clippy::float_cmp, clippy::approx_constant)]
    fn incremental_insert() {
        let points = (0..4)
            .map(|i| Point(i as isize, i as isize, 0))
            .collect::<Vec<_>>();
        let values = vec!["zero", "one", "two", "three"];
        let seed = ThreadRng::default().random::<u64>();
        let builder = Builder::default().seed(seed);

        let mut map = builder.build(points, values);

        map.insert_multiple(vec![Point(4, 4, 0)], vec!["four"]);

        let mut search = Search::default();

        for (i, item) in map.search(&Point(4, 4, 0), &mut search).enumerate() {
            match i {
                0 => {
                    assert_eq!(item.distance, 0.0);
                    assert_eq!(item.value, &"four");
                }
                1 => {
                    assert_eq!(item.distance, 1.4142135);
                    assert!(item.value == &"three");
                }
                2 => {
                    assert_eq!(item.distance, 2.828427);
                    assert!(item.value == &"two");
                }
                3 => {
                    assert_eq!(item.distance, 4.2426405);
                    assert!(item.value == &"one");
                }
                4 => {
                    assert_eq!(item.distance, 5.656854);
                    assert!(item.value == &"zero");
                }
                _ => unreachable!(),
            }
        }

        // Note
        // This has the same expected results as incremental_insert but builds
        // the whole map in one go. Only here for comparison.
        {
            let points = (0..5)
                .map(|i| Point(i as isize, i as isize, 0))
                .collect::<Vec<_>>();
            let values = vec!["zero", "one", "two", "three", "four"];
            let seed = ThreadRng::default().random::<u64>();
            let builder = Builder::default().seed(seed);
            let map = builder.build(points, values);
            let mut search = Search::default();
            for (i, item) in map.search(&Point(4, 4, 0), &mut search).enumerate() {
                match i {
                    0 => {
                        assert_eq!(item.distance, 0.0);
                        assert_eq!(item.value, &"four");
                    }
                    1 => {
                        assert_eq!(item.distance, 1.4142135);
                        assert!(item.value == &"three");
                    }
                    2 => {
                        assert_eq!(item.distance, 2.828427);
                        assert!(item.value == &"two");
                    }
                    3 => {
                        assert_eq!(item.distance, 4.2426405);
                        assert!(item.value == &"one");
                    }
                    4 => {
                        assert_eq!(item.distance, 5.656854);
                        assert!(item.value == &"zero");
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}
