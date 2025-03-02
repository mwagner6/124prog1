use core::f32;
use std::env;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use frand::Rand;

// DHeap struct
struct DHeap {
    heap: Vec<(f32, usize)>,
    d: usize,
}

// Function for determining the parent of an index in heap
fn heap_parent(i: usize, d: usize) -> usize {
    (i - 1) / d
}

// Function to determine children indices of an index in heap
fn heap_children(i: usize, d: usize) -> Vec<usize> {
    (1+(i*d)..3+(i*d)).collect()
}

impl DHeap {
    // Function to verify that parents of an item in heap are valid (recursive)
    fn verify_up(&mut self, i: usize) {
        // If not the root node, check that value is larger than parent. If not, swap
        if i != 0 {
            let p = heap_parent(i, self.d);
            if self.heap[i].0 < self.heap[p].0 {
                self.heap.swap(i, p);
                // Verify upwards from parent node
                self.verify_up(p);
            }
        }
    }

    // Functino to recursively verify that children of an item are valid
    fn verify_down(&mut self, i: usize) {
        let ci = heap_children(i, self.d);
        let mut swap = false;
        let mut  min = self.heap[i].0;
        let mut min_idx = i;
        // Loop over children, finding the child with the smallest value
        for idx in ci {
            if idx >= self.heap.len() {
                break
            }
            if self.heap[idx].0 < min {
                min_idx = idx;
                min = self.heap[idx].0;
                swap = true;
            }
        }
        // If we found a child with value smaller than current node's value, swap them and verify downard from the child
        if swap {
            self.heap.swap(i, min_idx);
            self.verify_down(min_idx);
        }
    }

    // Insert item into and verify parents
    fn push(&mut self, item: (f32, usize)) {
        self.heap.push(item);
        self.verify_up(self.heap.len()-1);
    }

    // Remove smallest item from heap and verify its children
    fn pop(&mut self) -> (f32, usize){
        if self.heap.len() < 1 {
            panic!("Tried to pop from empty heap");
        }
        let out = self.heap[0];
        // Remove first item in heap, and verify downwards from the root to ensure that the heap is still valid 
        if self.heap.len() > 1{
            self.heap[0] = self.heap.pop().unwrap();
            self.verify_down(0);
        } else {
            self.heap.pop();
        }
        out
    }
}

// Graph struct
#[derive(Debug)]
struct Graph {
    n: usize,
    adj: Vec<Vec<(usize, f32)>> 
}


impl Graph {
    // Initialize new graph with $n$ nodes, and a vector of adjacency lists
    fn new(n: usize) -> Graph {
        let map = vec![Vec::new(); n];
        Graph {n: n, adj: map}        
    }

    // Add one-directional edges (To be verified and copied over later)
    fn add_edge_one_way(&mut self, u: usize, v: usize, weight: f32) {
        self.adj[u].push((v, weight));
    }

    // Function to add many edges
    fn add_edges_batch_one_way(&mut self, edges: Vec<(usize, usize, f32)>) {
        for (u, v, weight) in edges {
            self.add_edge_one_way(u, v, weight);
        }
    }

    // For each edge, push its reverse edge in order to ensure our graph is symmetric (undirected)
    fn ensure_symmetry(&mut self) {
        let n = self.n;
        
        let current_adj = self.adj.clone();
        
        // For all edges, check that their reverse also exists
        for u in 0..n {
            for &(v, weight) in &current_adj[u] {
                let mut has_reverse = false;
                for &(w, _) in &self.adj[v] {
                    if w == u {
                        has_reverse = true;
                        break;
                    }
                }
                
                if !has_reverse {
                    self.adj[v].push((u, weight));
                }
            }
        }
    }

    // Function to return neighbors of a node
    fn neighbors(&self, u: usize) -> Vec<(usize, f32)>{
        self.adj[u].clone()
    }
}

// Prim's algorithm implementation
fn mst_prim(g: &Graph) -> f32 {
    // Initialize lists for distances and visited nodes
    let n = g.n;
    let mut visited = vec![false; n];
    let mut dists = vec![f32::INFINITY; n];

    let s = 0;
    dists[s] = 0.0;

    // Initialize our heap and add our source node to it
    let mut heap = DHeap { heap: vec![], d: 2 };
    heap.push((0.0, s));

    // While there are items in our heap, we remove the item with the minimum distance, and if it leads to an unvisited node, we visit that node
    while !heap.heap.is_empty() {
        let (_key, u) = heap.pop();
        if visited[u] {
            continue;
        }
        visited[u] = true;

        // We only keep track of the weight of added edges, not weight to reach them from the source, to make total weight easier to compute
        for (v, weight) in g.neighbors(u) {
            if !visited[v] && dists[v] > weight {
                dists[v] = weight;
                heap.push((weight, v));
            }
        }
    }

    dists.iter().sum()
}

// Generate complete basic graph of random weights with multithreading
fn complete_basic(n: usize) -> Graph {
    let threshold = 17.68 / n as f32;
    let g = Graph::new(n);
    let graph = Arc::new(Mutex::new(g));

    // Use parallel iterators to allocate work for each thread
    (0..n).into_par_iter().for_each(|u| {
        let mut local_edges = Vec::new();
        let mut rng = Rand::new();

        let graph_clone = Arc::clone(&graph);

        // Generate upper triangular edges, then later we will copy edges over
        for v in u + 1..n {
            let w = rng.gen::<f32>();
            if w < threshold {
                local_edges.push((u, v, w));
            }
        }

        // Insert edges in batches so we lock our Mutex less
        let mut g = graph_clone.lock().unwrap();
        g.add_edges_batch_one_way(local_edges);
    });

    // Ensure graph symmetry after all parallel work is done
    let mut graph = Arc::try_unwrap(graph).unwrap().into_inner().unwrap();
    graph.ensure_symmetry();
    graph
}

// Create hypercube graph of random weights via multithreading
fn hypercube(n: usize) -> Graph {
    let g = Graph::new(n);
    
    let graph = Arc::new(Mutex::new(g));
    
    (0..n).into_par_iter().for_each(|u| {
        let mut local_edges = Vec::new();
        let mut rng = Rand::new();

        let graph_clone = Arc::clone(&graph);

        for v in u+1..n {
            let diff = v-u;
            if diff & diff - 1 == 0 {
                local_edges.push((u, v, rng.gen::<f32>()));
            }
        }

        let mut g = graph_clone.lock().unwrap();
        g.add_edges_batch_one_way(local_edges);
    });
    
    let mut graph = Arc::try_unwrap(graph).unwrap().into_inner().unwrap();
    graph.ensure_symmetry();
    graph
}

// Fitted to 5.82*(n^-0.6)
// We can bound at 5.82*1.2 * (n^-0.6)
fn complete_unit_square(n: usize) -> Graph {
    let dropoutbound: f32 = 6.98 * (n as f32).powf(-0.6);
    
    // Generate locations first
    let mut rng = Rand::new();
    let locs: Vec<(f32, f32)> = (0..n).map(
        |_| (rng.gen::<f32>(), rng.gen::<f32>())
    ).collect();
    
    let g = Graph::new(n);
    let locs_arc = Arc::new(locs);
    let graph = Arc::new(Mutex::new(g));
    
    (0..n).into_par_iter().for_each(|u| {
        let mut local_edges = Vec::new();
        let graph_clone = Arc::clone(&graph);
        let dropoutbound_copy = dropoutbound;
        let locs_clone = Arc::clone(&locs_arc);

        for v in u+1..n {
            let w = ((locs_clone[u].0-locs_clone[v].0).powi(2)+(locs_clone[u].1-locs_clone[v].1).powi(2)).sqrt();
            if w < dropoutbound_copy {
                local_edges.push((u, v, w));
            }
        }

        let mut g = graph_clone.lock().unwrap();
        g.add_edges_batch_one_way(local_edges);
    });
    
    let mut graph = Arc::try_unwrap(graph).unwrap().into_inner().unwrap();
    graph.ensure_symmetry();
    graph
}
// Fitted to 3.71 * (n^-0.42)
// We can bound to 1.2*3.71 * (n^-0.42)
fn complete_unit_cube(n: usize) -> Graph {
    let dropoutbound: f32 = 4.45 * (n as f32).powf(-0.42);
    
    // Generate locations first
    let mut rng = Rand::new();
    let locs: Vec<(f32, f32, f32)> = (0..n).map(
        |_| (rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>())
    ).collect();
    
    let g = Graph::new(n);
    let locs_arc = Arc::new(locs);
    let graph = Arc::new(Mutex::new(g));
    
    (0..n).into_par_iter().for_each(|u| {
        let mut local_edges = Vec::new();
        let graph_clone = Arc::clone(&graph);
        let dropoutbound_copy = dropoutbound;
        let locs_clone = Arc::clone(&locs_arc);

        for v in u+1..n {
            let w = ((locs_clone[u].0-locs_clone[v].0).powi(2)+(locs_clone[u].1-locs_clone[v].1).powi(2)+(locs_clone[u].2-locs_clone[v].2).powi(2)).sqrt();
            if w < dropoutbound_copy {
                local_edges.push((u, v, w));
            }
        }

        let mut g = graph_clone.lock().unwrap();
        g.add_edges_batch_one_way(local_edges);
    });
    
    let mut graph = Arc::try_unwrap(graph).unwrap().into_inner().unwrap();
    graph.ensure_symmetry();
    graph
}

//Fit to 2.5 * (n^-0.28)
// We can bound to 1.2*2.5 * (n^-0.28)
fn complete_unit_hypercube(n: usize) -> Graph {
    let dropoutbound: f32 = 3.0 * (n as f32).powf(-0.28);
    
    // Generate locations first
    let mut rng = Rand::new();
    let locs: Vec<(f32, f32, f32, f32)> = (0..n).map(
        |_| (rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>())
    ).collect();
    
    let g = Graph::new(n);
    let locs_arc = Arc::new(locs);
    let graph = Arc::new(Mutex::new(g));
    
    (0..n).into_par_iter().for_each(|u| {
        let mut local_edges = Vec::new();
        let graph_clone = Arc::clone(&graph);
        let dropoutbound_copy = dropoutbound;
        let locs_clone = Arc::clone(&locs_arc);

        for v in u+1..n {
            let w = ((locs_clone[u].0-locs_clone[v].0).powi(2)+(locs_clone[u].1-locs_clone[v].1).powi(2)+(locs_clone[u].2-locs_clone[v].2).powi(2)+(locs_clone[u].3-locs_clone[v].3).powi(2)).sqrt();
            if w < dropoutbound_copy {
                local_edges.push((u, v, w));
            }
        }

        let mut g = graph_clone.lock().unwrap();
        g.add_edges_batch_one_way(local_edges);
    });
        
    let mut graph = Arc::try_unwrap(graph).unwrap().into_inner().unwrap();
    graph.ensure_symmetry();
    graph
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
    for _ in 0..numtrials {
        let g = generate_graph(dimension, numpoints);
        
        avgweight += mst_prim(&g);
    }
    avgweight = avgweight / numtrials as f32;
    println!("{avgweight} {numpoints} {numtrials} {dimension}");
}