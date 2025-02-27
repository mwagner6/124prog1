use core::{f64, time};
use std::{env, time::Duration};
use std::time::Instant;
use fastrand;

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

// Generate complete basic graph of random weights. Empirically,
// I fitted a curve over many trials to the maximum weight edge used.
// This fit the form k(n)=14.73/n. Conservatively, we can apply 
// double this, dropping out above 14.73*1.2/n.
fn complete_basic(n: usize) -> Graph {
    let mut g = Graph::new(n);
    for u in 0..n-1 {
        for v in u+1..n {
            let rand = fastrand::f64();
            if rand < 17.68 / n as f64 {
                g.add_edge(u, v, rand);
            }
        }
    }
    return g;
}

fn hypercube(n: usize) -> Graph {
    let mut g = Graph::new(n);
    for u in 0..n-1 {
        for v in u+1..n {
            let diff = v - u;
            if diff & diff - 1 == 0 {
                g.add_edge(u, v, fastrand::f64());
            }
        }
    }
    return g;
}

// Fitted to 5.82*(n^-0.6)
// We can bound at 5.82*1.2 * (n^-0.6)
fn complete_unit_square(n: usize) -> Graph {
    let dropoutbound: f64 = 6.98 * (n as f64).powf(-0.6);
    let mut g = Graph::new(n);
    let locs: Vec<(f64, f64)> = (0..n).map(
        |_| (fastrand::f64(), 
                fastrand::f64())
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
    let mut g = Graph::new(n);
    let locs: Vec<(f64, f64, f64)> = (0..n).map(
        |_| (fastrand::f64(), 
                fastrand::f64(), 
                fastrand::f64())
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
    let mut g = Graph::new(n);
    let locs: Vec<(f64, f64, f64, f64)> = (0..n).map(
        |_| (fastrand::f64(), 
                fastrand::f64(), 
                fastrand::f64(),
                fastrand::f64())
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
