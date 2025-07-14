use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{cmp::Ordering, fmt::Debug};

/// Mean radius of the Earth in meters
const EARTH_RADIUS_M: f32 = 6_371_000.0;
/// Constant to convert degrees to radians
const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;

/// A 2D coordinate
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coord<T: Copy + PartialOrd> {
    pub lat: T,
    pub lon: T,
}

impl<T> Serialize for Coord<T>
where
    T: Copy + PartialOrd + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let arr = [self.lat, self.lon];
        arr.serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for Coord<T>
where
    T: Copy + PartialOrd + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let arr = <[T; 2]>::deserialize(deserializer)?;
        Ok(Coord {
            lat: arr[0],
            lon: arr[1],
        })
    }
}

impl<T: Copy + PartialOrd> Coord<T> {
    pub fn new(lat: T, lon: T) -> Self {
        Self { lat, lon }
    }
}

impl<T: Copy + PartialOrd> std::ops::Index<usize> for Coord<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.lat,
            1 => &self.lon,
            _ => panic!("Coord index out of bounds: {idx}"),
        }
    }
}

/// A point in 2-dimensional space with attached data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point<T: Copy + PartialOrd, D> {
    pub coords: Coord<T>,
    pub data: D,
}

impl<T: Copy + PartialOrd, D> Point<T, D> {
    pub fn new(coords: Coord<T>, data: D) -> Self {
        Self { coords, data }
    }
}

/// BKD-tree node
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum BKDTree<T: Copy + PartialOrd, D> {
    Leaf(Vec<Point<T, D>>),
    Node {
        axis: usize,
        split: T,
        left: Box<BKDTree<T, D>>,
        right: Box<BKDTree<T, D>>,
        count: usize,
    },
}

impl<T: Copy + PartialOrd + Debug, D: Debug> Default for BKDTree<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + PartialOrd + Debug, D: Debug> BKDTree<T, D> {
    pub fn new() -> Self {
        BKDTree::Leaf(Vec::new())
    }

    pub fn len(&self) -> usize {
        match self {
            BKDTree::Leaf(points) => points.len(),
            BKDTree::Node { count, .. } => *count,
        }
    }

    pub fn insert(&mut self, point: Point<T, D>)
    where
        T: num_traits::Float,
    {
        println!("insert {:?}", point.coords);

        const LEAF_SIZE: usize = 16;
        match self {
            BKDTree::Leaf(ref mut points) => {
                points.reserve_exact(1);
                points.push(point);
                if points.len() > LEAF_SIZE {
                    // Split on axis with largest spread
                    let axis = find_largest_spread_axis(points);
                    points.sort_by(|a, b| {
                        a.coords[axis]
                            .partial_cmp(&b.coords[axis])
                            .unwrap_or(Ordering::Equal)
                    });
                    let median = points.len() / 2;
                    let split = points[median].coords[axis];
                    let right_points = points.split_off(median);
                    let left_points = std::mem::take(points);
                    let left_count = left_points.len();
                    let right_count = right_points.len();
                    *self = BKDTree::Node {
                        axis,
                        split,
                        left: Box::new(BKDTree::Leaf(left_points)),
                        right: Box::new(BKDTree::Leaf(right_points)),
                        count: left_count + right_count,
                    };
                }
            }
            BKDTree::Node {
                axis,
                split,
                left,
                right,
                count,
            } => {
                if point.coords[*axis] < *split {
                    left.insert(point);
                } else {
                    right.insert(point);
                }
                *count += 1;
            }
        }
    }

    // Iterator for radius query
    pub fn search_by_radius<'a>(
        &'a self,
        center: Coord<T>,
        radius: T,
        distance_fn: fn(&Coord<T>, &Coord<T>) -> T,
        inside: bool,
    ) -> RadiusQueryIter<'a, T, D>
    where
        T: num_traits::Float,
    {
        RadiusQueryIter {
            stack: vec![self],
            center,
            radius,
            buffer: Vec::new(),
            distance_fn,
            inside,
        }
    }

    pub fn search_by_polygon<'a>(
        &'a self,
        polygon: Vec<Coord<T>>,
        inside: bool,
    ) -> PolygonQueryIter<'a, T, D>
    where
        T: num_traits::Float,
    {
        PolygonQueryIter {
            stack: vec![self],
            polygon,
            inside,
            buffer: Vec::new(),
        }
    }

    pub fn iter(&self) -> BKDTreeIter<T, D> {
        BKDTreeIter {
            stack: vec![self],
            buffer: Vec::new(),
        }
    }

    /// Deletes all points whose data matches any element in the given set.
    /// This scans the entire tree and is inefficient (O(n)).
    pub fn delete(&mut self, data_to_delete: &std::collections::HashSet<D>)
    where
        D: PartialEq + Eq + std::hash::Hash,
    {
        match self {
            BKDTree::Leaf(ref mut points) => {
                points.retain(|p| !data_to_delete.contains(&p.data));
            }
            BKDTree::Node {
                left, right, count, ..
            } => {
                left.delete(data_to_delete);
                right.delete(data_to_delete);
                *count = left.len() + right.len();
            }
        }
    }
}

// Helper function to find the axis with the largest spread
fn find_largest_spread_axis<T: Copy + PartialOrd, D>(points: &[Point<T, D>]) -> usize
where
    T: num_traits::Float,
{
    let mut max_spread = T::zero();
    let mut axis = 0;
    for i in 0..2 {
        let min = points
            .iter()
            .map(|p| p.coords[i])
            .fold(T::max_value(), |a, b| a.min(b));
        let max = points
            .iter()
            .map(|p| p.coords[i])
            .fold(T::min_value(), |a, b| a.max(b));
        let spread = max - min;
        if spread > max_spread {
            max_spread = spread;
            axis = i;
        }
    }
    axis
}

impl<T: Copy + PartialOrd + std::fmt::Debug, D: std::fmt::Debug> BKDTree<T, D> {
    /// Recursively display the tree structure, showing splits and points.
    pub fn display(&self, depth: usize) {
        let indent = "  ".repeat(depth);
        match self {
            BKDTree::Leaf(points) => {
                println!("{}Leaf with {} points:", indent, points.len());
                for p in points {
                    println!(
                        "{}  Point: coords={:?}, data={:?}",
                        indent, p.coords, p.data
                    );
                }
            }
            BKDTree::Node {
                axis,
                split,
                left,
                right,
                ..
            } => {
                println!("{indent}Node: axis={axis}, split={split:?}");
                left.display(depth + 1);
                right.display(depth + 1);
            }
        }
    }
}

#[inline]
fn to_radians<T: num_traits::Float>(deg: T) -> f32 {
    let deg_f32: f32 = num_traits::cast::<T, f32>(deg).unwrap();
    deg_f32 * DEG_TO_RAD
}

/// Computes the Haversine distance (in meters) between two points (assumed to be [lat, lon] in degrees)
pub fn haversine_distance<T: num_traits::Float + Copy + PartialOrd + Debug>(
    coord1: &Coord<T>,
    coord2: &Coord<T>,
) -> f32 {
    let lat1 = to_radians(coord1.lat);
    let lon1 = to_radians(coord1.lon);
    let lat2 = to_radians(coord2.lat);
    let lon2 = to_radians(coord2.lon);
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_M * c
}

// Iterator for radius query
pub struct RadiusQueryIter<'a, T: Copy + PartialOrd, D> {
    stack: Vec<&'a BKDTree<T, D>>,
    center: Coord<T>,
    radius: T,
    buffer: Vec<&'a D>,
    distance_fn: fn(&Coord<T>, &Coord<T>) -> T,
    inside: bool,
}

impl<'a, T: num_traits::Float + Copy + PartialOrd + Debug, D> Iterator
    for RadiusQueryIter<'a, T, D>
{
    type Item = &'a D;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(data) = self.buffer.pop() {
                return Some(data);
            }
            let node = self.stack.pop()?;
            match node {
                BKDTree::Leaf(points) => {
                    for p in points.iter().rev() {
                        let dist = (self.distance_fn)(&self.center, &p.coords);
                        if (self.inside && dist <= self.radius)
                            || (!self.inside && dist > self.radius)
                        {
                            self.buffer.push(&p.data);
                        }
                    }
                }
                BKDTree::Node {
                    axis,
                    split,
                    left,
                    right,
                    ..
                } => {
                    let diff = self.center[*axis] - *split;
                    let abs_diff = diff.abs();
                    let sqrt_radius2 = self.radius.sqrt();
                    if diff <= T::zero() {
                        self.stack.push(left);
                        if abs_diff <= sqrt_radius2 {
                            self.stack.push(right);
                        }
                    } else {
                        self.stack.push(right);
                        if abs_diff <= sqrt_radius2 {
                            self.stack.push(left);
                        }
                    }
                }
            }
        }
    }
}

pub fn is_point_in_polygon<T: Copy + PartialOrd + num_traits::Float>(
    polygon: &[Coord<T>],
    point: &Coord<T>,
) -> bool {
    let mut is_inside = false;
    let lon = point.lon;
    let lat = point.lat;
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let xi = polygon[i].lon;
        let yi = polygon[i].lat;
        let xj = polygon[j].lon;
        let yj = polygon[j].lat;

        let intersect = (yi > lat) != (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi;
        if intersect {
            is_inside = !is_inside;
        }
        j = i;
    }
    is_inside
}

pub struct PolygonQueryIter<'a, T: Copy + PartialOrd, D> {
    stack: Vec<&'a BKDTree<T, D>>,
    polygon: Vec<Coord<T>>,
    inside: bool,
    buffer: Vec<&'a D>,
}

impl<'a, T: num_traits::Float + Copy + PartialOrd, D> Iterator for PolygonQueryIter<'a, T, D> {
    type Item = &'a D;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(data) = self.buffer.pop() {
                return Some(data);
            }
            let node = self.stack.pop()?;
            match node {
                BKDTree::Leaf(points) => {
                    for p in points.iter().rev() {
                        let inside = is_point_in_polygon(&self.polygon, &p.coords);
                        if (inside && self.inside) || (!inside && !self.inside) {
                            self.buffer.push(&p.data);
                        }
                    }
                }
                BKDTree::Node { left, right, .. } => {
                    self.stack.push(left);
                    self.stack.push(right);
                }
            }
        }
    }
}

// Iterator over all points in the BKDTree
pub struct BKDTreeIter<'a, T: Copy + PartialOrd, D> {
    stack: Vec<&'a BKDTree<T, D>>,
    buffer: Vec<&'a Point<T, D>>,
}

impl<'a, T: Copy + PartialOrd, D> Iterator for BKDTreeIter<'a, T, D> {
    type Item = &'a Point<T, D>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(point) = self.buffer.pop() {
                return Some(point);
            }
            let node = self.stack.pop()?;
            match node {
                BKDTree::Leaf(points) => {
                    // Push all points in reverse order so we can pop from the end
                    for p in points.iter().rev() {
                        self.buffer.push(p);
                    }
                }
                BKDTree::Node { left, right, .. } => {
                    self.stack.push(left);
                    self.stack.push(right);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use bincode;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn distance2<T: Copy + PartialOrd + num_traits::Float>(c1: &Coord<T>, c2: &Coord<T>) -> f32 {
        let dx = num_traits::cast::<T, f32>(c1.lat - c2.lat).unwrap();
        let dy = num_traits::cast::<T, f32>(c1.lon - c2.lon).unwrap();
        (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn test_insert_and_query_radius() {
        let mut tree = BKDTree::<f32, &'static str>::new();
        let pts = vec![
            Point {
                coords: Coord { lat: 0.0, lon: 0.0 },
                data: "a",
            },
            Point {
                coords: Coord { lat: 1.0, lon: 1.0 },
                data: "b",
            },
            Point {
                coords: Coord { lat: 2.0, lon: 2.0 },
                data: "c",
            },
            Point {
                coords: Coord { lat: 2.1, lon: 2.1 },
                data: "d",
            },
            Point {
                coords: Coord { lat: 3.0, lon: 3.0 },
                data: "e",
            },
            Point {
                coords: Coord { lat: 5.0, lon: 5.0 },
                data: "f",
            },
        ];
        for p in pts {
            tree.insert(p);
        }
        let center = Point {
            coords: Coord { lat: 1.0, lon: 1.0 },
            data: "b",
        };
        let found: Vec<&str> = tree
            .search_by_radius(center.coords, 1.5, distance2, true)
            .copied()
            .collect();
        // Should find "a", "b", "c"
        assert_eq!(found.len(), 3);
        assert!(found.contains(&"a"));
        assert!(found.contains(&"b"));
        assert!(found.contains(&"c"));
    }

    #[test]
    fn test_insert_same_coords_different_data() {
        let mut tree = BKDTree::<f32, &'static str>::new();
        let p1 = Point {
            coords: Coord { lat: 1.0, lon: 1.0 },
            data: "a",
        };
        let p2 = Point {
            coords: Coord { lat: 1.0, lon: 1.0 },
            data: "b",
        };
        let p3 = Point {
            coords: Coord { lat: 1.0, lon: 1.0 },
            data: "c",
        };
        tree.insert(p1);
        tree.insert(p2);
        tree.insert(p3);
        // Query with a small radius to get all points at [1.0, 1.0]
        let center = Point {
            coords: Coord { lat: 1.0, lon: 1.0 },
            data: "irrelevant",
        };
        let found: Vec<&str> = tree
            .search_by_radius(center.coords, 0.01, distance2, true)
            .copied()
            .collect();
        assert_eq!(found.len(), 3);
        assert!(found.contains(&"a"));
        assert!(found.contains(&"b"));
        assert!(found.contains(&"c"));
    }

    #[test]
    fn test_query_radius_iter() {
        let mut tree = BKDTree::<f32, &'static str>::new();
        let pts = vec![
            Point {
                coords: Coord { lat: 0.0, lon: 0.0 },
                data: "a",
            },
            Point {
                coords: Coord { lat: 1.0, lon: 1.0 },
                data: "b",
            },
            Point {
                coords: Coord { lat: 2.0, lon: 2.0 },
                data: "c",
            },
            Point {
                coords: Coord { lat: 2.1, lon: 2.1 },
                data: "d",
            },
            Point {
                coords: Coord { lat: 3.0, lon: 3.0 },
                data: "e",
            },
            Point {
                coords: Coord { lat: 5.0, lon: 5.0 },
                data: "f",
            },
        ];
        for p in pts {
            tree.insert(p);
        }
        let center = Point {
            coords: Coord { lat: 1.0, lon: 1.0 },
            data: "b",
        };
        let found: Vec<&str> = tree
            .search_by_radius(center.coords, 1.5, distance2, true)
            .copied()
            .collect();
        // Should find "a", "b", "c"
        assert_eq!(found.len(), 3);
        assert!(found.contains(&"a"));
        assert!(found.contains(&"b"));
        assert!(found.contains(&"c"));
    }

    #[test]
    fn test_display_with_random_points() {
        // Insert a lot of random points and display the tree
        const NUM_POINTS: usize = 100;
        let mut tree = BKDTree::<f32, usize>::new();
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..NUM_POINTS {
            let coords = Coord {
                lat: rng.random_range(0.0..100.0),
                lon: rng.random_range(0.0..100.0),
            };
            let point = Point { coords, data: i };
            tree.insert(point);
        }
        // Display the tree structure
        tree.display(0);
    }

    #[test]
    fn test_search_by_polygon_iter_triangle() {
        // Define a triangle (not a rectangle)
        let polygon = vec![
            Coord { lat: 0.0, lon: 0.0 },
            Coord { lat: 5.0, lon: 0.0 },
            Coord { lat: 2.5, lon: 5.0 },
        ];
        // Points: some inside, some outside
        let points = vec![
            (
                Point {
                    coords: Coord { lat: 2.5, lon: 2.0 },
                    data: "inside",
                },
                true,
            ),
            (
                Point {
                    coords: Coord { lat: 1.0, lon: 1.0 },
                    data: "inside2",
                },
                true,
            ),
            (
                Point {
                    coords: Coord { lat: 4.0, lon: 1.0 },
                    data: "inside3",
                },
                true,
            ),
            (
                Point {
                    coords: Coord { lat: 2.5, lon: 4.9 },
                    data: "near_top",
                },
                true,
            ),
            (
                Point {
                    coords: Coord { lat: 2.5, lon: 5.1 },
                    data: "above",
                },
                false,
            ),
            (
                Point {
                    coords: Coord { lat: 5.1, lon: 0.0 },
                    data: "right",
                },
                false,
            ),
            (
                Point {
                    coords: Coord {
                        lat: -1.0,
                        lon: 0.0,
                    },
                    data: "left",
                },
                false,
            ),
        ];
        let mut tree = BKDTree::<f32, &str>::new();
        for (p, _) in &points {
            tree.insert(p.clone());
        }

        let found: Vec<&str> = tree.search_by_polygon(polygon, true).copied().collect();
        for (p, should_be_inside) in &points {
            if *should_be_inside {
                assert!(
                    found.contains(&p.data),
                    "Point {:?} should be inside polygon",
                    p
                );
            } else {
                assert!(
                    !found.contains(&p.data),
                    "Point {:?} should be outside polygon",
                    p
                );
            }
        }
    }

    impl<T: Copy + PartialOrd + PartialEq, D: PartialEq> PartialEq for BKDTree<T, D> {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (BKDTree::Leaf(a), BKDTree::Leaf(b)) => a == b,
                (
                    BKDTree::Node {
                        axis: a_axis,
                        split: a_split,
                        left: a_left,
                        right: a_right,
                        count: a_count,
                    },
                    BKDTree::Node {
                        axis: b_axis,
                        split: b_split,
                        left: b_left,
                        right: b_right,
                        count: b_count,
                    },
                ) => {
                    a_axis == b_axis
                        && a_split == b_split
                        && a_left == b_left
                        && a_right == b_right
                        && a_count == b_count
                }
                _ => false,
            }
        }
    }

    #[test]
    fn test_serialize_deserialize_bkdtree() {
        let mut tree = BKDTree::<f32, usize>::new();
        for i in 0..10 {
            let coords = Coord {
                lat: i as f32,
                lon: (i * 2) as f32,
            };
            let point = Point { coords, data: i };
            tree.insert(point);
        }
        let encoded = bincode::serialize(&tree).expect("serialize BKDTree");
        let decoded: BKDTree<f32, usize> =
            bincode::deserialize(&encoded).expect("deserialize BKDTree");
        assert_eq!(tree, decoded);
    }

    #[test]
    fn test_test1() {
        let mut tree = BKDTree::<f32, &'static str>::new();
        let pts = vec![
            Point {
                coords: Coord {
                    lat: 9.0814233,
                    lon: 45.2623823,
                },
                data: "1",
            },
            Point {
                coords: Coord {
                    lat: 9.0979028,
                    lon: 45.1995182,
                },
                data: "2",
            },
        ];
        for p in pts {
            tree.insert(p);
        }
        let center = Point {
            coords: Coord {
                lat: 9.1418481,
                lon: 45.2324096,
            },
            data: "b",
        };
        let found: Vec<&str> = tree
            .search_by_radius(center.coords, 10_000.0, haversine_distance, true)
            .copied()
            .collect();
        // Should find "a", "b", "c"
        assert_eq!(found.len(), 2);
        assert!(found.contains(&"1"));
        assert!(found.contains(&"2"));
    }

    #[test]
    fn test_test2() {
        let mut tree = BKDTree::<f32, &'static str>::new();
        let pts = vec![
            Point::new(
                Coord {
                    lat: -72.1928787,
                    lon: 42.9309292,
                },
                "1",
            ),
            Point::new(
                Coord {
                    lat: -72.1928787,
                    lon: 42.929908,
                },
                "2",
            ),
            Point::new(
                Coord {
                    lat: -72.1912479,
                    lon: 42.9302222,
                },
                "3",
            ),
            Point::new(
                Coord {
                    lat: -72.1917844,
                    lon: 42.9312277,
                },
                "4",
            ),
            Point::new(
                Coord {
                    lat: -72.1928787,
                    lon: 42.9309292,
                },
                "5",
            ),
            Point::new(
                Coord {
                    lat: -10.2328721,
                    lon: 20.9385112,
                },
                "6",
            ),
        ];

        for p in pts {
            tree.insert(p);
        }
        let center = Coord {
            lat: -10.2328758,
            lon: 20.938517,
        };
        let found: HashSet<&str> = tree
            .search_by_radius(center, 10_000.0, haversine_distance, false)
            .copied()
            .collect();
        assert_eq!(found, HashSet::from(["1", "2", "3", "4", "5",]));
    }

    #[test]
    fn test_test3() {
        let mut tree = BKDTree::<f32, &'static str>::new();
        let pts = vec![
            Point::new(
                Coord {
                    lat: -50.6964111,
                    lon: 70.2120854,
                },
                "1",
            ),
            Point::new(
                Coord {
                    lat: -50.7403564,
                    lon: 70.1823094,
                },
                "2",
            ),
            Point::new(
                Coord {
                    lat: -51.2512207,
                    lon: 70.1123535,
                },
                "3",
            ),
            Point::new(
                Coord {
                    lat: -50.8639526,
                    lon: 70.0796264,
                },
                "4",
            ),
            Point::new(
                Coord {
                    lat: -50.6167603,
                    lon: 70.0973989,
                },
                "5",
            ),
        ];

        for p in pts {
            tree.insert(p);
        }

        let found: HashSet<&str> = tree
            .search_by_polygon(
                vec![
                    Coord {
                        lat: -51.3693237,
                        lon: 70.4082687,
                    },
                    Coord {
                        lat: -51.5643311,
                        lon: 69.8623282,
                    },
                    Coord {
                        lat: -49.9822998,
                        lon: 69.8273124,
                    },
                    Coord {
                        lat: -49.7543335,
                        lon: 70.3787763,
                    },
                    Coord {
                        lat: -51.3693237,
                        lon: 70.4082687,
                    },
                ],
                true,
            )
            .copied()
            .collect();
        assert_eq!(found, HashSet::from(["1", "2", "3", "4", "5",]));
    }

    #[test]
    fn test_len_empty_and_insert() {
        let mut tree = BKDTree::<f32, i32>::new();
        assert_eq!(tree.len(), 0);
        tree.insert(Point::new(Coord::new(1.0, 2.0), 42));
        assert_eq!(tree.len(), 1);
        tree.insert(Point::new(Coord::new(2.0, 3.0), 43));
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_len_after_delete() {
        let mut tree = BKDTree::<f32, i32>::new();
        for i in 0..5 {
            tree.insert(Point::new(Coord::new(i as f32, i as f32), i));
        }
        assert_eq!(tree.len(), 5);
        let mut to_delete = HashSet::new();
        to_delete.insert(2);
        to_delete.insert(66); // this doesn't exist
        to_delete.insert(4);
        tree.delete(&to_delete);
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_len_after_split() {
        let mut tree = BKDTree::<f32, i32>::new();
        // Insert more than LEAF_SIZE points to force a split
        for i in 0..20 {
            tree.insert(Point::new(Coord::new(i as f32, i as f32), i));
        }
        assert_eq!(tree.len(), 20);
    }

    #[test]
    fn test_search_by_radius_inside_and_outside() {
        let mut tree = BKDTree::<f32, &'static str>::new();
        let pts = vec![
            Point {
                coords: Coord { lat: 0.0, lon: 0.0 },
                data: "a",
            },
            Point {
                coords: Coord { lat: 1.0, lon: 1.0 },
                data: "b",
            },
            Point {
                coords: Coord { lat: 2.0, lon: 2.0 },
                data: "c",
            },
            Point {
                coords: Coord { lat: 5.0, lon: 5.0 },
                data: "d",
            },
        ];
        for p in pts {
            tree.insert(p);
        }
        let center = Coord { lat: 1.0, lon: 1.0 };
        let radius = 2.0f32;
        // inside = true: should return a, b, c
        let found_inside: Vec<&str> = tree
            .search_by_radius(center, radius, distance2, true)
            .copied()
            .collect();
        assert!(found_inside.contains(&"a"));
        assert!(found_inside.contains(&"b"));
        assert!(found_inside.contains(&"c"));
        assert!(!found_inside.contains(&"d"));
        // inside = false: should return only d
        let found_outside: Vec<&str> = tree
            .search_by_radius(center, radius, distance2, false)
            .copied()
            .collect();
        assert!(!found_outside.contains(&"a"));
        assert!(!found_outside.contains(&"b"));
        assert!(!found_outside.contains(&"c"));
        assert!(found_outside.contains(&"d"));
    }

    #[test]
    fn test_search_by_polygon_inside_and_outside() {
        // Define a square polygon
        let polygon = vec![
            Coord { lat: 0.0, lon: 0.0 },
            Coord { lat: 0.0, lon: 2.0 },
            Coord { lat: 2.0, lon: 2.0 },
            Coord { lat: 2.0, lon: 0.0 },
        ];
        let points = vec![
            (
                Point {
                    coords: Coord { lat: 1.0, lon: 1.0 },
                    data: "inside",
                },
                true,
            ),
            (
                Point {
                    coords: Coord { lat: 0.5, lon: 1.5 },
                    data: "inside2",
                },
                true,
            ),
            (
                Point {
                    coords: Coord { lat: 2.5, lon: 1.0 },
                    data: "outside1",
                },
                false,
            ),
            (
                Point {
                    coords: Coord {
                        lat: -1.0,
                        lon: 1.0,
                    },
                    data: "outside2",
                },
                false,
            ),
        ];
        let mut tree = BKDTree::<f32, &str>::new();
        for (p, _) in &points {
            tree.insert(p.clone());
        }
        // inside = true: should return only points inside the polygon
        let found_inside: Vec<&str> = tree
            .search_by_polygon(polygon.clone(), true)
            .copied()
            .collect();
        for (p, is_inside) in &points {
            if *is_inside {
                assert!(found_inside.contains(&p.data), "Should contain {}", p.data);
            } else {
                assert!(
                    !found_inside.contains(&p.data),
                    "Should not contain {}",
                    p.data
                );
            }
        }
        // inside = false: should return only points outside the polygon
        let found_outside: Vec<&str> = tree.search_by_polygon(polygon, false).copied().collect();
        for (p, is_inside) in &points {
            if !*is_inside {
                assert!(found_outside.contains(&p.data), "Should contain {}", p.data);
            } else {
                assert!(
                    !found_outside.contains(&p.data),
                    "Should not contain {}",
                    p.data
                );
            }
        }
    }
}
