use rand::rngs::ThreadRng;
use simsimd::SpatialSimilarity;
use std::cmp::{Ord, Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;

type DistanceFn = fn(&[f32], &[f32]) -> Option<f64>;
type Vector = Vec<f32>;

#[derive(Clone, Debug)]
struct SearchCandidate<K: Ord + Hash + Clone> {
    node: Box<LayerNode<K>>,
    distance: f64,
}

impl<K: Ord + Hash + Clone> PartialEq for SearchCandidate<K> {
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance)
    }
}

impl<K: Ord + Hash + Clone> Eq for SearchCandidate<K> {}

impl<K: Ord + Hash + Clone> PartialOrd for SearchCandidate<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<K: Ord + Hash + Clone> Ord for SearchCandidate<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone, Debug)]
struct Layer<K: Ord + Hash + Clone> {
    nodes: HashMap<K, Box<LayerNode<K>>>,
}

impl<K: Ord + Hash + Clone> Layer<K> {
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
struct Graph<K: Ord + Hash + Clone> {
    distance: DistanceFn,
    rng: ThreadRng,
    m: usize,
    ml: f64,
    ef_search: usize,
    layers: Vec<Layer<K>>,
}

impl<K: Ord + Hash + Clone> Graph<K> {
    fn new() -> Self {
        Self {
            distance: f32::cosine,
            rng: rand::SeedableRng::from_entropy(),
            m: 16,
            ml: 0.25,
            ef_search: 20,
            layers: Vec::new(),
        }
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
}

#[derive(Clone, Debug)]
struct Node<K: Ord> {
    key: K,
    value: Vector,
}

impl<K: Ord> Node<K> {
    fn new(key: K, value: Vector) -> Self {
        Node { key, value }
    }
}

#[derive(Clone, Debug)]
struct LayerNode<K: Ord + Hash + Clone> {
    node: Node<K>,
    neighbors: Option<HashMap<K, Box<LayerNode<K>>>>,
}

impl<K: Ord + Hash + Clone> LayerNode<K> {
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
        let mut candidates: BinaryHeap<Reverse<SearchCandidate<K>>> =
            BinaryHeap::with_capacity(ef_search);
        let mut result: BinaryHeap<Reverse<SearchCandidate<K>>> = BinaryHeap::with_capacity(k);
        let mut visited: HashMap<K, bool> = HashMap::new();

        let initial_candidate = SearchCandidate {
            node: Box::new(self.clone()),
            distance: distance_fn(&self.node.value, target).unwrap_or(f64::INFINITY),
        };

        candidates.push(Reverse(initial_candidate.clone()));
        result.push(Reverse(initial_candidate.clone()));
        visited.insert(self.node.key.clone(), true);

        while let Some(Reverse(current_candidate)) = candidates.pop() {
            let mut improved = false;

            if let Some(ref neighbors) = current_candidate.node.neighbors {
                let mut neighbor_keys: Vec<&K> = neighbors.keys().collect();
                neighbor_keys.sort();

                for neighbor_key in neighbor_keys {
                    if visited.contains_key(&neighbor_key) {
                        continue;
                    }
                    visited.insert(neighbor_key.clone(), true);

                    if let Some(neighbor) = neighbors.get(&neighbor_key) {
                        let distance =
                            distance_fn(&neighbor.node.value, target).unwrap_or(f64::INFINITY);

                        if let Some(Reverse(current_best)) = result.peek() {
                            improved = improved || distance < current_best.distance;
                        }

                        let candidate = SearchCandidate {
                            node: Box::new((**neighbor).clone()),
                            distance,
                        };

                        if result.len() < k {
                            result.push(Reverse(candidate.clone()));
                        } else if let Some(Reverse(worst)) = result.peek() {
                            if distance < worst.distance {
                                result.pop();
                                result.push(Reverse(candidate.clone()));
                            }
                        }

                        candidates.push(Reverse(candidate));
                        if candidates.len() >= ef_search {
                            candidates.pop();
                        }
                    }

                    candidates.push(Reverse(current_candidate.clone()));
                    if candidates.len() >= ef_search {
                        candidates.pop();
                    }
                }
            }

            if !improved && result.len() >= k {
                break;
            }
        }

        result
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(candidate)| candidate)
            .collect()
    }

    fn add_neighbor(&mut self, new_node: Box<LayerNode<K>>, m: usize, distance_fn: DistanceFn) {
        let neighbors = self
            .neighbors
            .get_or_insert_with(|| HashMap::with_capacity(m));

        neighbors.insert(new_node.node.key.clone(), new_node);
        if neighbors.len() <= m {
            return;
        }

        let mut worst_dist = f64::NEG_INFINITY;
        let mut worst_key = None;

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

        let mut candidates = Vec::new();
        for neighbor in neighbors.values() {
            if let Some(neighbor_neighbors) = &neighbor.neighbors {
                for (key, candidate) in neighbor_neighbors {
                    if neighbors.contains_key(key) {
                        continue;
                    }
                    if *key == self.node.key {
                        continue;
                    }
                    candidates.push((key.clone(), candidate.clone()));
                }
            }
        }

        for (key, candidate) in candidates {
            neighbors.insert(key, Box::new((*candidate).clone()));
            if neighbors.len() >= m {
                break;
            }
        }
    }
}
