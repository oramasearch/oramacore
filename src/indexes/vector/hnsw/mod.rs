use anyhow::Result;
use rand::rngs::ThreadRng;
use rand::Rng;
use simsimd::SpatialSimilarity;
use std::cmp::{Ord, Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

type DistanceFn = fn(&[f32], &[f32]) -> Option<f64>;
type Vector = Vec<f32>;

#[derive(Clone, Debug)]
struct SearchCandidate<K: Ord + Hash + Clone + Debug> {
    node: Box<LayerNode<K>>,
    distance: f64,
}

impl<K: Ord + Hash + Clone + Debug> Ord for SearchCandidate<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl<K: Ord + Hash + Clone + Debug> Eq for SearchCandidate<K> {}

impl<K: Ord + Hash + Clone + Debug> PartialOrd for SearchCandidate<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord + Hash + Clone + Debug> PartialEq for SearchCandidate<K> {
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance)
    }
}

#[derive(Clone, Debug)]
struct Layer<K: Ord + Hash + Clone + Debug> {
    nodes: HashMap<K, Box<LayerNode<K>>>,
}

impl<K: Ord + Hash + Clone + Debug> Layer<K> {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    fn entry(&self) -> Option<&LayerNode<K>> {
        self.nodes.values().next().map(|n| &**n)
    }

    fn size(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Debug)]
struct Graph<K: Ord + Hash + Clone + Debug> {
    distance: DistanceFn,
    rng: ThreadRng,
    m: usize,
    ml: f64,
    ef_search: usize,
    layers: Vec<Layer<K>>,
    size: usize,
}

impl<K: Ord + Hash + Clone + Debug> Graph<K> {
    pub fn new() -> Self {
        Self {
            distance: |a, b| f32::cosine(a, b).map(|sim| 1.0 - sim),
            rng: rand::thread_rng(),
            m: 16,
            ml: 0.25,
            ef_search: 20,
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

    fn add(&mut self, nodes: Vec<Node<K>>) -> Result<()> {
        for node in nodes {
            let key = node.key.clone();
            let vec = node.value.clone();
            let insert_level = self.random_level()?;

            while insert_level >= self.layers.len() {
                self.layers.push(Layer::new());
            }

            if insert_level < 0 {
                anyhow::bail!("Invalid level")
            }

            let mut elevator: Option<Box<K>> = None;
            let pre_len = self.len();

            for i in (0..=self.layers.len() - 1).rev() {
                let mut layer = &mut self.layers[i];
                let mut new_node = Box::new(LayerNode {
                    node: Node {
                        key: key.clone(),
                        value: vec.clone(),
                    },
                    neighbors: Some(HashMap::new()),
                });

                if layer.entry().is_none() {
                    let mut nodes = HashMap::new();
                    nodes.insert(key.clone(), new_node);
                    layer.nodes = nodes;
                    continue;
                }

                let search_point = if let Some(elevator_key) = &elevator {
                    if let Some(node) = layer.nodes.get(&**elevator_key) {
                        node
                    } else {
                        layer.entry().unwrap()
                    }
                } else {
                    layer.entry().unwrap()
                };

                let neighborhood = search_point.search(self.m, self.ef_search, &vec, self.distance);

                if neighborhood.is_empty() {
                    anyhow::bail!("No nodes found");
                }

                elevator = Some(Box::new(neighborhood[0].node.node.key.clone()));

                if insert_level >= i {
                    if layer.nodes.contains_key(&key) {
                        self.delete(&key);
                        layer = &mut self.layers[i];
                    }

                    layer.nodes.insert(key.clone(), new_node.clone());

                    for candidate in neighborhood {
                        let node = layer.nodes.get_mut(&candidate.node.node.key).unwrap();
                        node.add_neighbor(new_node.clone(), self.m, self.distance);
                        new_node.add_neighbor(Box::new((**node).clone()), self.m, self.distance);
                    }
                }
            }

            self.size += 1;
            if self.len() != pre_len + 1 {
                anyhow::bail!("Node not added")
            }
        }

        Ok(())
    }

    pub fn delete(&mut self, key: &K) -> bool {
        if self.layers.is_empty() {
            return false;
        }

        let mut deleted = false;
        let m = self.m;

        for layer in &mut self.layers {
            if let Some(mut node) = layer.nodes.remove(key) {
                node.isolate(m);
                deleted = true;
            }

            for (_, other_node) in layer.nodes.iter_mut() {
                if let Some(ref mut neighbors) = other_node.neighbors {
                    neighbors.remove(key);
                }
            }
        }

        if deleted {
            self.size -= 1;
        }

        deleted
    }

    pub fn search(&self, near: &[f32], k: usize) -> Vec<Node<K>> {
        if self.layers.is_empty() {
            return Vec::new();
        }

        let mut elevator: Option<Box<K>> = None;

        for layer_idx in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_idx];

            let search_point = if let Some(elevator_key) = &elevator {
                layer.nodes.get(&**elevator_key).unwrap()
            } else {
                layer.entry().unwrap()
            };

            if layer_idx > 0 {
                let nodes = search_point.search(1, self.ef_search, near, self.distance);
                elevator = Some(Box::new(nodes[0].node.node.key.clone()));
                continue;
            }

            let nodes = search_point.search(k, self.ef_search, near, self.distance);
            return nodes
                .into_iter()
                .map(|candidate| candidate.node.node.clone())
                .collect();
        }

        unreachable!("Graph search should never reach this point")
    }

    fn random_level(&mut self) -> Result<usize> {
        let mut max = 1;
        if self.layers.len() > 0 {
            if self.ml == 0.0 {
                anyhow::bail!("ml must be greater than 0.0");
            }

            max = max_level(self.ml, self.layers[0].size())?;
        }

        let mut level = 0;
        while level < max {
            let random: f64 = self.rng.gen();
            if random > self.ml {
                return Ok(level);
            }

            level += 1;
        }

        Ok(max)
    }

    fn dims(&self) -> usize {
        if self.layers.is_empty() {
            return 0;
        }

        self.layers[0]
            .entry()
            .map(|node| node.node.value.len())
            .unwrap_or(0)
    }

    fn ptr(v: K) -> Box<K> {
        Box::new(v)
    }
}

#[derive(Clone, Debug)]
struct Node<K: Ord + Debug> {
    key: K,
    value: Vector,
}

impl<K: Ord + Debug> Node<K> {
    fn new(key: K, value: Vector) -> Self {
        Node { key, value }
    }
}

#[derive(Clone, Debug)]
struct LayerNode<K: Ord + Hash + Clone + Debug> {
    node: Node<K>,
    neighbors: Option<HashMap<K, Box<LayerNode<K>>>>,
}

impl<K: Ord + Hash + Clone + Debug> LayerNode<K> {
    pub fn new(key: K, value: Vector) -> Self {
        Self {
            node: Node::new(key, value),
            neighbors: Some(HashMap::new()),
        }
    }

    fn search(
        &self,
        k: usize,
        ef_search: usize,
        target: &[f32],
        distance_fn: DistanceFn,
    ) -> Vec<SearchCandidate<K>> {
        let mut candidates: BinaryHeap<SearchCandidate<K>> = BinaryHeap::with_capacity(ef_search);
        let mut result: BinaryHeap<SearchCandidate<K>> = BinaryHeap::with_capacity(k);
        let mut visited: HashMap<K, bool> = HashMap::new();

        let initial_candidate = SearchCandidate {
            node: Box::new(self.clone()),
            distance: distance_fn(&self.node.value, target).unwrap_or(f64::INFINITY),
        };

        candidates.push(initial_candidate.clone());
        result.push(initial_candidate);
        visited.insert(self.node.key.clone(), true);

        while let Some(current_candidate) = candidates.pop() {
            if let Some(worst_in_result) = result.peek() {
                if current_candidate.distance > worst_in_result.distance && result.len() >= k {
                    break;
                }
            }

            if let Some(ref neighbors) = current_candidate.node.neighbors {
                for (neighbor_key, neighbor) in neighbors {
                    if visited.contains_key(neighbor_key) {
                        continue;
                    }
                    visited.insert(neighbor_key.clone(), true);

                    let distance =
                        distance_fn(&neighbor.node.value, target).unwrap_or(f64::INFINITY);
                    let candidate = SearchCandidate {
                        node: Box::new((**neighbor).clone()),
                        distance,
                    };

                    if result.len() < k {
                        result.push(candidate.clone());
                    } else if let Some(worst) = result.peek() {
                        if distance < worst.distance {
                            result.pop();
                            result.push(candidate.clone());
                        }
                    }

                    if candidates.len() < ef_search {
                        candidates.push(candidate);
                    }
                }
            }
        }

        result.into_sorted_vec()
    }

    fn isolate(&mut self, m: usize) {
        if let Some(ref mut self_neighbors) = self.neighbors {
            let neighbor_keys: Vec<K> = self_neighbors.keys().cloned().collect();

            for neighbor_key in neighbor_keys {
                if let Some(neighbor) = self_neighbors.get_mut(&neighbor_key) {
                    if let Some(ref mut neighbor_neighbors) = neighbor.neighbors {
                        neighbor_neighbors.remove(&self.node.key);
                        neighbor.replenish(m);
                    }
                }
            }
            self_neighbors.clear();
        }
    }

    fn add_neighbor(&mut self, new_node: Box<LayerNode<K>>, m: usize, distance_fn: DistanceFn) {
        let neighbors = self
            .neighbors
            .get_or_insert_with(|| HashMap::with_capacity(m));
        neighbors.insert(new_node.node.key.clone(), new_node);

        if neighbors.len() <= m {
            return;
        }

        let mut worst_key: Option<K> = None;
        let mut worst_dist = f64::NEG_INFINITY;

        for (key, neighbor) in neighbors.iter() {
            let dist =
                distance_fn(&neighbor.node.value, &self.node.value).unwrap_or(f64::NEG_INFINITY);
            if dist > worst_dist || worst_key.is_none() {
                worst_dist = dist;
                worst_key = Some(key.clone());
            }
        }

        if let Some(worst) = worst_key {
            if let Some(mut removed) = neighbors.remove(&worst) {
                if let Some(removed_neighbors) = &mut removed.neighbors {
                    removed_neighbors.remove(&self.node.key);
                }
                removed.replenish(m);
            }
        }
    }

    fn replenish(&mut self, m: usize) {
        let Some(neighbors) = &mut self.neighbors else {
            return;
        };

        if neighbors.len() >= m {
            return;
        }

        let mut candidates: Vec<(K, Box<LayerNode<K>>)> = Vec::new();

        for neighbor in neighbors.values() {
            if let Some(neighbor_neighbors) = &neighbor.neighbors {
                for (key, candidate) in neighbor_neighbors {
                    if neighbors.contains_key(key) || *key == self.node.key {
                        continue;
                    }
                    candidates.push((key.clone(), candidate.clone()));
                }
            }
        }

        for (key, candidate) in candidates {
            neighbors.insert(key, candidate);
            if neighbors.len() >= m {
                break;
            }
        }
    }
}

pub fn max_level(ml: f64, num_nodes: usize) -> Result<usize> {
    if ml == 0.0 {
        anyhow::bail!("ml must be greater than 0");
    }

    if num_nodes == 0 {
        return Ok(1);
    }

    let level = (num_nodes as f64).ln();
    let level = level / (1.0 / ml).ln();

    Ok((level.round() as usize) + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_graph() -> Graph<i32> {
        let mut graph = Graph::new();
        graph
            .add(vec![
                Node::new(1, vec![1.0, 0.0]),
                Node::new(2, vec![0.0, 1.0]),
                Node::new(3, vec![1.0, 1.0]),
                Node::new(4, vec![0.0, 0.0]),
            ])
            .unwrap();
        graph
    }

    #[test]
    fn test_add_nodes() {
        let mut graph = Graph::new();
        assert_eq!(graph.len(), 0);

        graph
            .add(vec![
                Node::new(1, vec![1.0, 0.0]),
                Node::new(2, vec![0.0, 1.0]),
            ])
            .unwrap();
        assert_eq!(graph.len(), 2);

        graph.add(vec![Node::new(3, vec![1.0, 1.0])]).unwrap();
        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn test_search() {
        let graph = create_test_graph();

        let results = graph.search(&[0.5, 0.5], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, 3);

        let results = graph.search(&[0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, 4);
        assert_eq!(results[1].key, 2);
    }

    #[test]
    fn test_delete_node() {
        let mut graph = create_test_graph();
        assert_eq!(graph.len(), 4);

        assert!(graph.delete(&3));
        assert_eq!(graph.len(), 3);

        let results = graph.search(&[1.0, 1.0], 1);
        assert_eq!(results.len(), 1);
        assert_ne!(results[0].key, 3);
    }

    #[test]
    fn test_search_edge_cases() {
        let mut graph = Graph::new();

        let results = graph.search(&[0.5, 0.5], 1);
        assert!(results.is_empty());

        graph.add(vec![Node::new(1, vec![1.0, 0.0])]).unwrap();
        let results = graph.search(&[0.5, 0.5], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, 1);

        let results = graph.search(&[0.5, 0.5], 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_random_level() {
        let mut graph = Graph::new();
        graph.ml = 0.5;

        for _ in 0..100 {
            let level = graph.random_level().unwrap();
            assert!(level >= 0 && level <= 1);
        }

        graph.add(vec![Node::new(1, vec![1.0, 0.0])]).unwrap();
        graph.add(vec![Node::new(2, vec![0.0, 1.0])]).unwrap();

        for _ in 0..100 {
            let level = graph.random_level().unwrap();
            assert!(level >= 0 && level <= 2);
        }
    }
}
