use core::{f64, time};
use std::{collections::HashMap, env, hash::Hash, time::Duration};
use rand::{distr::{Distribution, Uniform}, Rng};
use std::time::Instant;


struct DHeap {
    d: usize,
    heap: Vec<(f64, usize)>,
    index_map: HashMap<usize, usize>, // HashMap to track the index of each usize in the heap
}

fn heap_parent(i: usize, d: usize) -> usize {
    (i - 1) / d
}

fn heap_children(i: usize, d: usize) -> Vec<usize> {
    (1 + (i * d)..3 + (i * d)).collect()
}

impl DHeap {
    fn verify_up(&mut self, i: usize) {
        if i != 0 {
            let p = heap_parent(i, self.d);
            if self.heap[i].0 < self.heap[p].0 {
                self.heap.swap(i, p);
                self.index_map.insert(self.heap[i].1, i);
                self.index_map.insert(self.heap[p].1, p);
                self.verify_up(p);
            }
        }
    }

    fn verify_down(&mut self, i: usize) {
        let ci = heap_children(i, self.d);
        let mut swap = false;
        let mut min = self.heap[i].0;
        let mut min_idx = i;
        for idx in ci {
            if idx >= self.heap.len() {
                break;
            }
            if self.heap[idx].0 < min {
                min_idx = idx;
                min = self.heap[idx].0;
                swap = true;
            }
        }
        if swap {
            self.heap.swap(i, min_idx);
            self.index_map.insert(self.heap[i].1, i);
            self.index_map.insert(self.heap[min_idx].1, min_idx);
            self.verify_down(min_idx);
        }
    }

    fn push(&mut self, item: (f64, usize)) {
        // If the item with the same usize exists, update if the new weight is smaller
        if let Some(&idx) = self.index_map.get(&item.1) {
            if self.heap[idx].0 > item.0 {
                self.heap[idx] = item;
                self.verify_up(idx);
            }
        } else {
            // If it's a new usize, simply push it
            self.heap.push(item);
            let idx = self.heap.len() - 1;
            self.index_map.insert(item.1, idx);
            self.verify_up(idx);
        }
    }

    fn pop(&mut self) -> (f64, usize) {
        if self.heap.len() < 1 {
            panic!("Tried to pop from empty heap");
        }
        let out = self.heap[0];
        if self.heap.len() > 1 {
            let last = self.heap.pop().unwrap();
            self.heap[0] = last;
            self.index_map.insert(last.1, 0); // Update the index of the last item
            self.index_map.remove(&out.1); // Remove the index of the popped item
            self.verify_down(0);
        } else {
            self.heap.pop();
            self.index_map.remove(&out.1); // Remove the index of the popped item
        }
        out
    }
}

struct Graph {
    n: usize,
    adj: HashMap<usize, Vec<(usize, f64)>>
}

impl Graph {
    fn new(n: usize) -> Graph {
        let mut map = HashMap::new();
        for i in 0..n {
            map.insert(i, Vec::with_capacity(n));
        }
        Graph {n: n, adj: map}        
    }

    fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        self.adj.get_mut(&u).unwrap().push((v, weight));
        self.adj.get_mut(&v).unwrap().push((u, weight));
    }

    fn neighbors(&self, u: usize) -> Vec<(usize, f64)>{
        self.adj.get(&u).unwrap().to_vec()
    }
}

fn mst_prim(g: &Graph) -> (f64, f64, Vec<(usize, usize, f64)>) {
    let n = g.n;
    let mut visited = vec![false; n];
    let mut dists = vec![f64::INFINITY; n];
    let mut parent = vec![None; n]; // Track where each node was reached from
    let mut edges = Vec::with_capacity(n - 1);

    let mut max_edge_weight = 0.0;
    let mut total_weight = 0.0;

    let s = 0;
    dists[s] = 0.0;
    let mut heap = DHeap { d: n,heap: vec![] , index_map: HashMap::new()};
    heap.push((0.0, s));

    while !heap.heap.is_empty() {
        let (_key, u) = heap.pop();
        if visited[u] {
            continue;
        }
        visited[u] = true;

        // If u has a parent, add the edge to the MST
        if let Some(p) = parent[u] {
            let weight = dists[u];
            edges.push((p, u, weight));
            total_weight += weight;
            if weight > max_edge_weight {
                max_edge_weight = weight;
            }
        }

        for (v, weight) in g.neighbors(u) {
            if !visited[v] && dists[v] > weight {
                dists[v] = weight;
                parent[v] = Some(u); // Track parent of v
                heap.push((weight, v));
            }
        }
    }

    (total_weight, max_edge_weight, edges)
}

// Generate complete basic graph of random weights. Empirically,
// I fitted a curve over many trials to the maximum weight edge used.
// This fit the form k(n)=14.73/n. Conservatively, we can apply 
// double this, dropping out above 14.73*1.2/n.
fn complete_basic(n: usize) -> Graph {
    let mut rng = rand::rng();
    let range = Uniform::new(0.0_f64, 1.0_f64).unwrap();
    let mut g = Graph::new(n);
    for u in 0..n-1 {
        for v in u+1..n {
            let rand = range.sample(&mut rng);
            if rand < 17.68 / n as f64 {
                g.add_edge(u, v, rand);
            }
        }
    }
    return g;
}

fn hypercube(n: usize) -> Graph {
    let mut rng = rand::rng();
    let mut g = Graph::new(n);
    for u in 0..n-1 {
        for v in u+1..n {
            let diff = v - u;
            if diff & diff - 1 == 0 {
                g.add_edge(u, v, rng.random_range(0.0_f64..1.0));
            }
        }
    }
    return g;
}

// Fitted to 5.82*(n^-0.6)
// We can bound at 5.82*1.2 * (n^-0.6)
fn complete_unit_square(n: usize) -> Graph {
    let dropoutbound: f64 = 6.98 * (n as f64).powf(-0.6);
    let mut rng = rand::rng();
    let mut g = Graph::new(n);
    let locs: Vec<(f64, f64)> = (0..n).map(
        |_| (rng.random_range(0.0_f64..1.0), 
                rng.random_range(0.0_f64..1.0))
        ).collect();
    for u in 0..n-1 {
        for v in u+1..n {
            let w = ((locs[u].0-locs[v].0).powi(2)+
            (locs[u].1-locs[v].1).powi(2)).sqrt();
            if w < dropoutbound {
                g.add_edge(u, v, w);
            }
        }
    }
    return g;
}
// Fitted to 3.71 * (n^-0.42)
// We can bound to 1.2*3.71 * (n^-0.42)
fn complete_unit_cube(n: usize) -> Graph {
    let dropoutbound: f64 = 4.45 * (n as f64).powf(-0.42);
    let mut rng = rand::rng();
    let mut g = Graph::new(n);
    let locs: Vec<(f64, f64, f64)> = (0..n).map(
        |_| (rng.random_range(0.0_f64..1.0), 
                rng.random_range(0.0_f64..1.0), 
                rng.random_range(0.0_f64..1.0))
        ).collect();
    for u in 0..n-1 {
        for v in u+1..n {
            let w = ((locs[u].0-locs[v].0).powi(2)+
            (locs[u].1-locs[v].1).powi(2)+
            (locs[u].2-locs[v].2).powi(2)).sqrt();
            if w < dropoutbound {
                g.add_edge(u, v, w);
            }
        }
    }
    return g;
}

//Fit to 2.5 * (n^-0.28)
// We can bound to 1.2*2.5 * (n^-0.28)
fn complete_unit_hypercube(n: usize) -> Graph {
    let dropoutbound: f64 = 3.0 * (n as f64).powf(-0.28);
    let mut rng = rand::rng();
    let mut g = Graph::new(n);
    let locs: Vec<(f64, f64, f64, f64)> = (0..n).map(
        |_| (rng.random_range(0.0_f64..1.0), 
                rng.random_range(0.0_f64..1.0), 
                rng.random_range(0.0_f64..1.0),
                rng.random_range(0.0_f64..1.0))
        ).collect();
    for u in 0..n-1 {
        for v in u+1..n {
            let w = ((locs[u].0-locs[v].0).powi(2)+
            (locs[u].1-locs[v].1).powi(2)+
            (locs[u].2-locs[v].2).powi(2)+
            (locs[u].3-locs[v].3).powi(2)).sqrt();
            if w < dropoutbound {
                g.add_edge(u, v, w);
            }
        }
    }
    return g;
}

fn generate_graph(dim: u32, n: u32) -> Graph {
    match dim {
        0 => {
            complete_basic(n as usize)
        },
        1 => {
            hypercube(n as usize)
        },
        2 => {
            complete_unit_square(n as usize)
        },
        3 => {
            complete_unit_cube(n as usize)
        },
        4 => {
            complete_unit_hypercube(n as usize)
        }
        _ => panic!("Invalid dimension")
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let numpoints: u32 = args[2].parse::<u32>().unwrap();
    let numtrials: u32 = args[3].parse::<u32>().unwrap();
    let dimension: u32 = args[4].parse::<u32>().unwrap();

    let mut avgweight = 0.0;
    let mut generation: time::Duration = Duration::ZERO;
    let mut mst: time::Duration = Duration::ZERO;
    let mut maxweight = 0.0;
    for _ in 0..numtrials {
        let genstart = Instant::now();
        let g = generate_graph(dimension, numpoints);
        generation += genstart.elapsed();
        let mststart = Instant::now();
        let out = mst_prim(&g);
        if out.1 > maxweight {
            maxweight = out.1;
        }
        avgweight += out.0;
        mst += mststart.elapsed();
    }
    avgweight = avgweight / numtrials as f64;
    println!("{avgweight} {numpoints} {numtrials} {dimension}");
    println!("Generation time: {generation:.2?}. MST time: {mst:.2?}");
    println!("Maximum weight edge used: {maxweight}");
}
