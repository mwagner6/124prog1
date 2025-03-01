use core::{f64, time};
use std::{env, time::Duration};
use std::time::Instant;
use fastrand;
use std::thread;
use std::sync::{Arc, Mutex};
use num_cpus;

#[derive(Debug)]
struct DHeap {
    heap: Vec<(f64, usize)>,
    d: usize,
}

fn heap_parent(i: usize, d: usize) -> usize {
    (i - 1) / d
}

fn heap_children(i: usize, d: usize) -> Vec<usize> {
    (1+(i*d)..3+(i*d)).collect()
}

impl DHeap {
    fn verify_up(&mut self, i: usize) {
        if i != 0 {
            let p = heap_parent(i, self.d);
            if self.heap[i].0 < self.heap[p].0 {
                self.heap.swap(i, p);
                self.verify_up(p);
            }
        }
    }

    fn verify_down(&mut self, i: usize) {
        let ci = heap_children(i, self.d);
        let mut swap = false;
        let mut  min = self.heap[i].0;
        let mut min_idx = i;
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
        if swap {
            self.heap.swap(i, min_idx);
            self.verify_down(min_idx);
        }
    }

    fn push(&mut self, item: (f64, usize)) {
        self.heap.push(item);
        self.verify_up(self.heap.len()-1);
    }

    fn pop(&mut self) -> (f64, usize){
        if self.heap.len() < 1 {
            panic!("Tried to pop from empty heap");
        }
        let out = self.heap[0];
        if self.heap.len() > 1{
            self.heap[0] = self.heap.pop().unwrap();
            self.verify_down(0);
        } else {
            self.heap.pop();
        }
        out
    }
}

#[derive(Debug)]
struct Graph {
    n: usize,
    adj: Vec<Vec<(usize, f64)>> 
}

impl Graph {
    fn new(n: usize) -> Graph {
        let map = vec![Vec::new(); n];
        Graph {n: n, adj: map}        
    }

    fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        self.adj[u].push((v, weight));
        self.adj[v].push((u, weight));
    }

    fn add_edges_batch(&mut self, edges: Vec<(usize, usize, f64)>) {
        for (u, v, weight) in edges {
            self.adj[u].push((v, weight));
            self.adj[v].push((u, weight));
        }
    }

    fn neighbors(&self, u: usize) -> Vec<(usize, f64)>{
        self.adj[u].clone()
    }
}

fn mst_prim(g: &Graph) -> f64 {
    let n = g.n;
    let mut visited = vec![false; n];
    let mut dists = vec![f64::INFINITY; n];

    let s = 0;
    dists[s] = 0.0;
    let mut heap = DHeap { heap: vec![], d: n };
    heap.push((0.0, s));

    while !heap.heap.is_empty() {
        let (_key, u) = heap.pop();
        if visited[u] {
            continue;
        }
        visited[u] = true;

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
    let num_threads = num_cpus::get();
    let mut g = Graph::new(n);
    let threshold = 17.68 / n as f64;
    
    // Create thread-local RNGs
    let mut handles = vec![];
    let graph = Arc::new(Mutex::new(g));
    
    for thread_id in 0..num_threads {
        let graph_clone = Arc::clone(&graph);
        let threshold_copy = threshold;
        let n_copy = n;
        
        let handle = thread::spawn(move || {
            let mut local_edges = Vec::new();
            let mut rng = fastrand::Rng::new();
            
            // Each thread processes a range of vertices
            let chunk_size = (n_copy - 1) / num_threads + 1;
            let start_u = thread_id * chunk_size;
            let end_u = std::cmp::min(start_u + chunk_size, n_copy - 1);
            
            for u in start_u..end_u {
                for v in u+1..n_copy {
                    let rand = rng.f64();
                    if rand < threshold_copy {
                        local_edges.push((u, v, rand));
                    }
                }
            }
            
            let mut g = graph_clone.lock().unwrap();
            g.add_edges_batch(local_edges);
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    Arc::try_unwrap(graph).unwrap().into_inner().unwrap()
}

fn hypercube(n: usize) -> Graph {
    let num_threads = num_cpus::get();
    let mut g = Graph::new(n);
    
    let mut handles = vec![];
    let graph = Arc::new(Mutex::new(g));
    
    for thread_id in 0..num_threads {
        let graph_clone = Arc::clone(&graph);
        let n_copy = n;
        
        let handle = thread::spawn(move || {
            let mut local_edges = Vec::new();
            let mut rng = fastrand::Rng::new();
            
            // Each thread processes a range of vertices
            let chunk_size = (n_copy - 1) / num_threads + 1;
            let start_u = thread_id * chunk_size;
            let end_u = std::cmp::min(start_u + chunk_size, n_copy - 1);
            
            for u in start_u..end_u {
                for v in u+1..n_copy {
                    let diff = v - u;
                    if diff & diff - 1 == 0 {
                        local_edges.push((u, v, rng.f64()));
                    }
                }
            }
            
            let mut g = graph_clone.lock().unwrap();
            g.add_edges_batch(local_edges);
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    Arc::try_unwrap(graph).unwrap().into_inner().unwrap()
}

// Fitted to 5.82*(n^-0.6)
// We can bound at 5.82*1.2 * (n^-0.6)
fn complete_unit_square(n: usize) -> Graph {
    let num_threads = num_cpus::get();
    let dropoutbound: f64 = 6.98 * (n as f64).powf(-0.6);
    
    // Generate locations first
    let mut rng = fastrand::Rng::new();
    let locs: Vec<(f64, f64)> = (0..n).map(
        |_| (rng.f64(), rng.f64())
    ).collect();
    
    let mut g = Graph::new(n);
    let locs_arc = Arc::new(locs);
    let mut handles = vec![];
    let graph = Arc::new(Mutex::new(g));
    
    for thread_id in 0..num_threads {
        let graph_clone = Arc::clone(&graph);
        let locs_clone = Arc::clone(&locs_arc);
        let n_copy = n;
        let dropoutbound_copy = dropoutbound;
        
        let handle = thread::spawn(move || {
            let mut local_edges = Vec::new();
            
            // Each thread processes a range of vertices
            let chunk_size = (n_copy - 1) / num_threads + 1;
            let start_u = thread_id * chunk_size;
            let end_u = std::cmp::min(start_u + chunk_size, n_copy - 1);
            
            for u in start_u..end_u {
                for v in u+1..n_copy {
                    let w = ((locs_clone[u].0-locs_clone[v].0).powi(2)+
                             (locs_clone[u].1-locs_clone[v].1).powi(2)).sqrt();
                    if w < dropoutbound_copy {
                        local_edges.push((u, v, w));
                    }
                }
            }
            
            let mut g = graph_clone.lock().unwrap();
            g.add_edges_batch(local_edges);
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    Arc::try_unwrap(graph).unwrap().into_inner().unwrap()
}
// Fitted to 3.71 * (n^-0.42)
// We can bound to 1.2*3.71 * (n^-0.42)
fn complete_unit_cube(n: usize) -> Graph {
    let num_threads = num_cpus::get();
    let dropoutbound: f64 = 4.45 * (n as f64).powf(-0.42);
    
    // Generate locations first
    let mut rng = fastrand::Rng::new();
    let locs: Vec<(f64, f64, f64)> = (0..n).map(
        |_| (rng.f64(), rng.f64(), rng.f64())
    ).collect();
    
    let mut g = Graph::new(n);
    let locs_arc = Arc::new(locs);
    let mut handles = vec![];
    let graph = Arc::new(Mutex::new(g));
    
    for thread_id in 0..num_threads {
        let graph_clone = Arc::clone(&graph);
        let locs_clone = Arc::clone(&locs_arc);
        let n_copy = n;
        let dropoutbound_copy = dropoutbound;
        
        let handle = thread::spawn(move || {
            let mut local_edges = Vec::new();
            
            // Each thread processes a range of vertices
            let chunk_size = (n_copy - 1) / num_threads + 1;
            let start_u = thread_id * chunk_size;
            let end_u = std::cmp::min(start_u + chunk_size, n_copy - 1);
            
            for u in start_u..end_u {
                for v in u+1..n_copy {
                    let w = ((locs_clone[u].0-locs_clone[v].0).powi(2)+
                             (locs_clone[u].1-locs_clone[v].1).powi(2)+
                             (locs_clone[u].2-locs_clone[v].2).powi(2)).sqrt();
                    if w < dropoutbound_copy {
                        local_edges.push((u, v, w));
                    }
                }
            }
            
            let mut g = graph_clone.lock().unwrap();
            g.add_edges_batch(local_edges);
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    Arc::try_unwrap(graph).unwrap().into_inner().unwrap()
}

//Fit to 2.5 * (n^-0.28)
// We can bound to 1.2*2.5 * (n^-0.28)
fn complete_unit_hypercube(n: usize) -> Graph {
    let num_threads = num_cpus::get();
    let dropoutbound: f64 = 3.0 * (n as f64).powf(-0.28);
    
    // Generate locations first
    let mut rng = fastrand::Rng::new();
    let locs: Vec<(f64, f64, f64, f64)> = (0..n).map(
        |_| (rng.f64(), rng.f64(), rng.f64(), rng.f64())
    ).collect();
    
    let mut g = Graph::new(n);
    let locs_arc = Arc::new(locs);
    let mut handles = vec![];
    let graph = Arc::new(Mutex::new(g));
    
    for thread_id in 0..num_threads {
        let graph_clone = Arc::clone(&graph);
        let locs_clone = Arc::clone(&locs_arc);
        let n_copy = n;
        let dropoutbound_copy = dropoutbound;
        
        let handle = thread::spawn(move || {
            let mut local_edges = Vec::new();
            
            // Each thread processes a range of vertices
            let chunk_size = (n_copy - 1) / num_threads + 1;
            let start_u = thread_id * chunk_size;
            let end_u = std::cmp::min(start_u + chunk_size, n_copy - 1);
            
            for u in start_u..end_u {
                for v in u+1..n_copy {
                    let w = ((locs_clone[u].0-locs_clone[v].0).powi(2)+
                             (locs_clone[u].1-locs_clone[v].1).powi(2)+
                             (locs_clone[u].2-locs_clone[v].2).powi(2)+
                             (locs_clone[u].3-locs_clone[v].3).powi(2)).sqrt();
                    if w < dropoutbound_copy {
                        local_edges.push((u, v, w));
                    }
                }
            }
            
            let mut g = graph_clone.lock().unwrap();
            g.add_edges_batch(local_edges);
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    Arc::try_unwrap(graph).unwrap().into_inner().unwrap()
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
    for _ in 0..numtrials {
        let genstart = Instant::now();
        let g = generate_graph(dimension, numpoints);
        generation += genstart.elapsed();
        let mststart = Instant::now();
        avgweight += mst_prim(&g);
        mst += mststart.elapsed();
    }
    avgweight = avgweight / numtrials as f64;
    println!("{avgweight} {numpoints} {numtrials} {dimension}");
    println!("Generation time: {generation:.2?}. MST time: {mst:.2?}");
}