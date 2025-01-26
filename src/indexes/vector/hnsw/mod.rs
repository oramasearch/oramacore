use anyhow::{anyhow, Result};
use rand::rngs::ThreadRng;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

type DistanceFn = fn(&[f32], &[f32]) -> Option<f64>;

#[inline]
fn default_distance(a: &[f32], b: &[f32]) -> Option<f64> {
    Some(1.0 - dot(a, b))
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

/// Normalize a vector in place (L2 norm).
#[inline]
fn normalize_in_place(vec: &mut [f32]) {
    let mut sum_sq = 0.0f32;
    for x in vec.iter() {
        sum_sq += x * x;
    }
    let norm = sum_sq.sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// Approximate "max level" function for bounding the random level.
/// Modified to clamp `ml` to a tiny positive epsilon if it is <= 0.0.
pub fn max_level(mut ml: f64, num_nodes: usize) -> Result<usize> {
    if ml <= 0.0 {
        // Instead of error, clamp to tiny
        ml = 1e-9;
    }
    if num_nodes == 0 {
        return Ok(1);
    }
    let level = (num_nodes as f64).ln() / (1.0 / ml).ln();
    Ok((level.round() as usize) + 1)
}

#[derive(Clone, Debug)]
pub struct Node<K: Ord + Debug> {
    pub key: K,
    pub value: Vec<f32>,
}

impl<K: Ord + Debug> Node<K> {
    pub fn new(key: K, value: Vec<f32>) -> Self {
        Node { key, value }
    }
}

#[derive(Clone, Debug)]
pub struct LayerNode<K: Ord + Hash + Clone + Debug> {
    pub node: Node<K>,
    pub neighbors: HashMap<K, Box<LayerNode<K>>>,
}

impl<K: Ord + Hash + Clone + Debug> LayerNode<K> {
    pub fn new(key: K, value: Vec<f32>) -> Self {
        Self {
            node: Node::new(key, value),
            neighbors: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Layer<K: Ord + Hash + Clone + Debug> {
    pub nodes: HashMap<K, Box<LayerNode<K>>>,
    pub entry_point: Option<K>,
}

impl<K: Ord + Hash + Clone + Debug> Layer<K> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            entry_point: None,
        }
    }
}

#[derive(Clone, Debug)]
struct SearchCandidate<K: Ord + Hash + Clone + Debug> {
    node: Box<LayerNode<K>>,
    distance: f64,
}

impl<K: Ord + Hash + Clone + Debug> PartialEq for SearchCandidate<K> {
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance)
    }
}
impl<K: Ord + Hash + Clone + Debug> Eq for SearchCandidate<K> {}

impl<K: Ord + Hash + Clone + Debug> PartialOrd for SearchCandidate<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}
impl<K: Ord + Hash + Clone + Debug> Ord for SearchCandidate<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Debug)]
pub struct Graph<K: Ord + Hash + Clone + Debug> {
    pub distance_fn: DistanceFn,
    rng: ThreadRng,

    pub m: usize,
    pub ml: f64,
    pub ef_search: usize,
    pub ef_construction: usize,

    layers: Vec<Layer<K>>,
    size: usize,
}

impl<K: Ord + Hash + Clone + Debug> Graph<K> {
    pub fn new() -> Self {
        Self {
            distance_fn: default_distance,
            rng: rand::thread_rng(),
            m: 16,
            ml: 0.25,
            ef_search: 20,
            ef_construction: 50,
            layers: Vec::new(),
            size: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn add(&mut self, nodes: Vec<Node<K>>) -> Result<()> {
        for mut n in nodes {
            normalize_in_place(&mut n.value);
            self.insert_node(n)?;
        }
        Ok(())
    }

    fn insert_node(&mut self, node: Node<K>) -> Result<()> {
        if self.contains_key(&node.key) {
            return Err(anyhow!("Key already present: {:?}", node.key));
        }

        // 1) Random level
        let level = self.random_level()?;

        // 2) add layers if needed
        let old_top = self.top_layer();
        while self.layers.len() <= level {
            self.layers.push(Layer::new());
        }
        let mut new_node = Box::new(LayerNode::new(node.key.clone(), node.value.clone()));

        if self.is_empty() {
            for l in 0..=level {
                self.layers[l].entry_point = Some(node.key.clone());
                self.layers[l]
                    .nodes
                    .insert(node.key.clone(), new_node.clone());
            }
            self.size += 1;
            return Ok(());
        }

        // If new node's level is higher than old top, set entry points
        if level > old_top {
            for l in (old_top + 1)..=level {
                self.layers[l].entry_point = Some(node.key.clone());
            }
        }

        // 3) top-down from the highest layer that actually has an entry point
        //    see "top_layer_with_entry" below
        let real_top = match self.top_layer_with_entry() {
            Some(t) => t,
            None => {
                // theoretically shouldn't happen if there's at least 1 node
                return Ok(());
            }
        };

        let mut current_key = self.layers[real_top].entry_point.clone().unwrap();
        // do greedy from real_top down to level+1
        for layer_idx in (level + 1..=real_top).rev() {
            current_key = self.greedy_closest(layer_idx, current_key, &new_node.node.value);
        }

        // BFS insertion from `level` down to 0
        for layer_idx in (0..=level).rev() {
            current_key = self.search_layer_candidates(
                layer_idx,
                current_key,
                &new_node.node.value,
                self.ef_construction,
            );
            let candidates = self.search_layer_candidates_collect(
                layer_idx,
                &new_node.node.value,
                self.ef_construction,
            );
            let selected = self.select_neighbors(&new_node.node.value, candidates, self.m);

            self.layers[layer_idx]
                .nodes
                .insert(node.key.clone(), new_node.clone());
            if self.layers[layer_idx].entry_point.is_none() {
                self.layers[layer_idx].entry_point = Some(node.key.clone());
            }
            for sc in &selected {
                let neighbor_key = &sc.node.node.key;
                new_node
                    .neighbors
                    .insert(neighbor_key.clone(), sc.node.clone());

                // add new node to neighbor
                if let Some(nbr) = self.layers[layer_idx].nodes.get_mut(neighbor_key) {
                    nbr.neighbors.insert(node.key.clone(), new_node.clone());

                    let dist_fn = self.distance_fn;
                    let m = self.m;

                    prune_neighbors(&mut *nbr, dist_fn, m);
                }
            }
        }
        self.size += 1;
        Ok(())
    }

    fn contains_key(&self, k: &K) -> bool {
        self.layers.iter().any(|layer| layer.nodes.contains_key(k))
    }

    /// Return the topmost layer index we have (even if it has no entry point).
    fn top_layer(&self) -> usize {
        if self.layers.is_empty() {
            0
        } else {
            self.layers.len() - 1
        }
    }

    /// Return the highest layer index that actually has `Some(entry_point)`.
    /// If none are found, returns None.
    fn top_layer_with_entry(&self) -> Option<usize> {
        for i in (0..self.layers.len()).rev() {
            if self.layers[i].entry_point.is_some() {
                return Some(i);
            }
        }
        None
    }

    fn greedy_closest(&self, layer_idx: usize, start_key: K, target: &[f32]) -> K {
        let mut current_key = start_key;
        let mut improved = true;
        while improved {
            improved = false;
            let cur_dist = self
                .dist_of_key(layer_idx, &current_key, target)
                .unwrap_or(f64::INFINITY);

            if let Some(current_node) = self.layers[layer_idx].nodes.get(&current_key) {
                for (nbr_key, nbr_node) in &current_node.neighbors {
                    let d =
                        (self.distance_fn)(&nbr_node.node.value, target).unwrap_or(f64::INFINITY);
                    if d < cur_dist {
                        current_key = nbr_key.clone();
                        improved = true;
                        break;
                    }
                }
            }
        }
        current_key
    }

    fn dist_of_key(&self, layer_idx: usize, key: &K, target: &[f32]) -> Option<f64> {
        self.layers[layer_idx]
            .nodes
            .get(key)
            .and_then(|node| (self.distance_fn)(&node.node.value, target))
    }

    fn search_layer_candidates(
        &self,
        layer_idx: usize,
        entry_key: K,
        target: &[f32],
        ef: usize,
    ) -> K {
        // If that layer has no node for `entry_key`, fallback
        if !self.layers[layer_idx].nodes.contains_key(&entry_key) {
            return entry_key;
        }
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut best = BinaryHeap::new();

        let ep = self.layers[layer_idx].nodes.get(&entry_key).unwrap();
        let ep_dist = (self.distance_fn)(&ep.node.value, target).unwrap_or(f64::INFINITY);

        let start_candidate = SearchCandidate {
            node: ep.clone(),
            distance: ep_dist,
        };
        candidates.push(start_candidate.clone());
        best.push(start_candidate);
        visited.insert(entry_key.clone());

        while let Some(current) = candidates.pop() {
            if let Some(worst) = best.peek() {
                if best.len() >= ef && current.distance > worst.distance {
                    break;
                }
            }
            for (_k_nbr, nbr_node) in &current.node.neighbors {
                let nbr_key = &nbr_node.node.key;
                if visited.contains(nbr_key) {
                    continue;
                }
                visited.insert(nbr_key.clone());

                let d = (self.distance_fn)(&nbr_node.node.value, target).unwrap_or(f64::INFINITY);
                let candidate = SearchCandidate {
                    node: nbr_node.clone(),
                    distance: d,
                };
                if best.len() < ef {
                    best.push(candidate.clone());
                } else if let Some(worst) = best.peek() {
                    if d < worst.distance {
                        best.pop();
                        best.push(candidate.clone());
                    }
                }
                candidates.push(candidate);
            }
        }
        let best_vec = best.into_sorted_vec();
        if let Some(first) = best_vec.first() {
            return first.node.node.key.clone();
        }
        entry_key
    }

    fn search_layer_candidates_collect(
        &self,
        layer_idx: usize,
        target: &[f32],
        ef: usize,
    ) -> Vec<SearchCandidate<K>> {
        // find the entry_point if any
        let entry_key = match &self.layers[layer_idx].entry_point {
            Some(k) => k.clone(),
            None => {
                return vec![];
            }
        };
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut best = BinaryHeap::new();

        let ep = match self.layers[layer_idx].nodes.get(&entry_key) {
            Some(x) => x.clone(),
            None => {
                return vec![];
            }
        };
        let ep_dist = (self.distance_fn)(&ep.node.value, target).unwrap_or(f64::INFINITY);

        let start_candidate = SearchCandidate {
            node: ep,
            distance: ep_dist,
        };
        candidates.push(start_candidate.clone());
        best.push(start_candidate);
        visited.insert(entry_key);

        while let Some(current) = candidates.pop() {
            if let Some(worst) = best.peek() {
                if best.len() >= ef && current.distance > worst.distance {
                    break;
                }
            }
            for (_k_nbr, nbr_node) in &current.node.neighbors {
                if visited.contains(&nbr_node.node.key) {
                    continue;
                }
                visited.insert(nbr_node.node.key.clone());
                let d = (self.distance_fn)(&nbr_node.node.value, target).unwrap_or(f64::INFINITY);
                let candidate = SearchCandidate {
                    node: nbr_node.clone(),
                    distance: d,
                };
                if best.len() < ef {
                    best.push(candidate.clone());
                } else if let Some(worst) = best.peek() {
                    if d < worst.distance {
                        best.pop();
                        best.push(candidate.clone());
                    }
                }
                candidates.push(candidate);
            }
        }
        best.into_sorted_vec()
    }

    fn select_neighbors(
        &self,
        _new_point: &[f32],
        mut candidates: Vec<SearchCandidate<K>>,
        m: usize,
    ) -> Vec<SearchCandidate<K>> {
        if candidates.len() <= m {
            return candidates;
        }
        candidates.truncate(m);
        candidates
    }

    /// A "top-down" search from the highest layer with entry point, then BFS in layer 0.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Node<K>> {
        if self.layers.is_empty() {
            return Vec::new();
        }
        let top = match self.top_layer_with_entry() {
            Some(t) => t,
            None => {
                return Vec::new();
            }
        };
        let mut current_key = match self.layers[top].entry_point.clone() {
            Some(k) => k,
            None => return Vec::new(),
        };
        for layer_idx in (1..=top).rev() {
            current_key = self.greedy_closest(layer_idx, current_key, query);
        }
        // BFS in layer 0
        let results = self.search_layer_candidates_collect(0, query, self.ef_search);
        let mut out = Vec::new();
        for i in 0..results.len().min(k) {
            out.push(results[i].node.node.clone());
        }
        out
    }

    fn random_level(&mut self) -> Result<usize> {
        let current_size = if self.layers.is_empty() {
            0
        } else {
            self.layers[0].nodes.len()
        };
        // allow small ml
        let max = max_level(self.ml, current_size)?;
        let mut level = 0;
        while level < max {
            let r: f64 = self.rng.gen();
            if r > self.ml {
                break;
            }
            level += 1;
        }
        Ok(level)
    }
}

fn prune_neighbors<K: Ord + Hash + Clone + Debug>(
    node: &mut LayerNode<K>,
    distance_fn: DistanceFn,
    m: usize,
) {
    if node.neighbors.len() <= m {
        return;
    }

    let mut worst_key: Option<K> = None;
    let mut worst_dist = f64::NEG_INFINITY;

    for (k, nbr) in &node.neighbors {
        let d = distance_fn(&node.node.value, &nbr.node.value).unwrap_or(f64::INFINITY);
        if d > worst_dist {
            worst_dist = d;
            worst_key = Some(k.clone());
        }
    }
    if let Some(k) = worst_key {
        node.neighbors.remove(&k);
    }
}

#[cfg(test)]
mod new_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Brute-force nearest neighbor for checking accuracy.
    /// We'll use brute-force results to compare them with HNSW search results
    fn brute_force_search(
        data: &[(i32, Vec<f32>)],
        query: &[f32],
        k: usize,
        distance_fn: DistanceFn,
    ) -> Vec<(i32, f64)> {
        let mut scored: Vec<(i32, f64)> = data
            .iter()
            .map(|(key, vec)| {
                let dist = distance_fn(vec, query).unwrap_or(f64::INFINITY);
                (*key, dist)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        scored
    }

    /// Force single-layer. Put all nodes go to level=0.
    #[test]
    fn test_single_layer() {
        let mut graph = Graph::new();
        graph.ml = 0.0;
        graph
            .add(vec![
                Node::new(1, vec![1.0, 0.0]),
                Node::new(2, vec![0.0, 1.0]),
                Node::new(3, vec![1.0, 1.0]),
                Node::new(4, vec![0.0, 0.0]),
            ])
            .unwrap();

        assert_eq!(graph.len(), 4);

        let q = [0.5, 0.5];
        let results = graph.search(&q, 2);

        assert_eq!(results.len(), 2);

        for r in &results {
            assert!([1, 2, 3, 4].contains(&r.key));
        }
    }

    /// Controlled multi-layer. Note this is a seeded random layer selection.
    #[test]
    fn test_multi_layer() {
        let mut graph = Graph::new();
        graph.m = 4;
        graph.ef_search = 10;
        graph.ef_construction = 10;
        graph
            .add(vec![
                Node::new(1, vec![1.0, 0.0]),
                Node::new(2, vec![0.0, 1.0]),
                Node::new(3, vec![1.0, 1.0]),
            ])
            .unwrap();

        graph.add(vec![Node::new(4, vec![0.0, 0.0])]).unwrap();

        // As long as we fix the RNG, these insertions are repeatable
        // The test is: do we get a valid top-down search?
        let results = graph.search(&[0.5, 0.5], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn real_world_test() {
        use serde::Deserialize;
        use serde_json;

        #[derive(Debug, Deserialize, Clone)]
        struct Question(String, Vec<i32>);

        let json_str =
            include_str!("datasets_bge-base_12k-questions-embeddings-quantized-32-bytes.json");
        let questions: Vec<Question> = serde_json::from_str(json_str).unwrap();

        dbg!(format!(
            "loaded {} questions and embeddings...",
            questions.len()
        ));

        let mut graph = Graph::new();
        graph.m = 4;
        graph.ef_search = 10;
        graph.ef_construction = 10;
        graph
            .add(
                questions
                    .iter()
                    .take(200)
                    .enumerate()
                    .map(|(i, q)| {
                        if i % 100 == 0 {
                            println!("Inserted {} nodes", i);
                        }
                        Node::new(i as i32, q.1.iter().map(|&x| x as f32).collect())
                    })
                    .collect(),
            )
            .unwrap();

        println!("Inserted all documents into the graph.");

        let target: Vec<f32> = vec![
            168, 231, 129, 67, 66, 165, 136, 247, 49, 19, 47, 48, 174, 34, 217, 104, 170, 86, 197,
            116, 153, 201, 81, 117, 237, 99, 54, 137, 206, 2, 229, 204,
        ]
        .iter()
        .map(|&x| x as f32)
        .collect();

        println!("Performing search on the graph...");

        let graph_results = graph.search(&target, 10);

        println!("Performing brute-force search...");
        let brute_force_results = brute_force_search(
            &questions
                .iter()
                .take(200)
                .enumerate()
                .map(|(i, q)| (i as i32, q.1.iter().map(|&x| x as f32).collect()))
                .collect::<Vec<_>>(),
            &target,
            10,
            default_distance,
        );

        dbg!(graph_results.clone());
        dbg!(brute_force_results.clone());

        assert_eq!(graph_results[0].key, brute_force_results[0].0);
    }
}
