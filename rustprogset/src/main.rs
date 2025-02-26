use core::f64;
use std::{collections::HashMap, env};
use rand::Rng;
use std::time::Instant;


struct BinHeap {
    heap: Vec<(f64, usize)>
}

fn heap_parent(i: usize) -> usize {
    (i - 1) / 2
}

fn heap_children(i: usize) -> Vec<usize> {
    (1+(i*2)..3+(i*2)).collect()
}

impl BinHeap {
    fn verify_up(&mut self, i: usize) {
        if i != 0 {
            let p = heap_parent(i);
            if self.heap[i].0 < self.heap[p].0 {
                self.heap.swap(i, p);
                self.verify_up(p);
            }
        }
    }

    fn verify_down(&mut self, i: usize) {
        let ci = heap_children(i);
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
    adj: HashMap<usize, Vec<(usize, f64)>>
}

impl Graph {
    fn new(n: usize) -> Graph {
        let mut map = HashMap::new();
        for i in 0..n {
            map.insert(i, vec![]);
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

fn mst_prim(g: Graph) -> f64 {
    let n = g.n;
    let mut visited = vec![false; n];
    let mut dists = vec![f64::INFINITY; n];

    let s = 0;
    dists[s] = 0.0;
    let mut heap = BinHeap{heap: vec![]};
    heap.push((0.0, s));

    while heap.heap.len() > 0 {
        let (_key, u) = heap.pop();
        if visited[u]{
            continue;
        }
        visited[u] = true;
        for (v, weight) in g.neighbors(u) {
            if (!visited[v]) && dists[v] > weight {
                dists[v] = weight;
                heap.push((weight, v))
            }
        }
    }
    return dists.iter().sum()
}

fn complete_basic(n: usize) -> Graph {
    let mut rng = rand::rng();
    let mut g = Graph::new(n);
    for u in 0..n-1 {
        for v in u+1..n {
            g.add_edge(u, v, rng.random_range(0.0_..1.0));
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

fn complete_unit_square(n: usize) -> Graph {
    let mut rng = rand::rng();
    let mut g = Graph::new(n);
    let locs: Vec<(f64, f64)> = (0..n).map(
        |_| (rng.random_range(0.0_f64..1.0), 
                rng.random_range(0.0_f64..1.0))
        ).collect();
    for u in 0..n-1 {
        for v in u+1..n {
            g.add_edge(u, v, 
                ((locs[u].0-locs[v].0).powi(2)+
                        (locs[u].1-locs[v].1).powi(2)).sqrt()
            );
        }
    }
    return g;
}

fn complete_unit_cube(n: usize) -> Graph {
    let mut rng = rand::rng();
    let mut g = Graph::new(n);
    let locs: Vec<(f64, f64, f64)> = (0..n).map(
        |_| (rng.random_range(0.0_f64..1.0), 
                rng.random_range(0.0_f64..1.0), 
                rng.random_range(0.0_f64..1.0))
        ).collect();
    for u in 0..n-1 {
        for v in u+1..n {
            g.add_edge(u, v, 
                ((locs[u].0-locs[v].0).powi(2)+
                        (locs[u].1-locs[v].1).powi(2)+
                        (locs[u].2-locs[v].2).powi(2)).sqrt()
            );
        }
    }
    return g;
}

fn complete_unit_hypercube(n: usize) -> Graph {
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
            g.add_edge(u, v, 
                ((locs[u].0-locs[v].0).powi(2)+
                        (locs[u].1-locs[v].1).powi(2)+
                        (locs[u].2-locs[v].2).powi(2)+
                        (locs[u].3-locs[v].3).powi(2)).sqrt()
            );
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
    let now = Instant::now();
    let args: Vec<String> = env::args().collect();
    let numpoints: u32 = args[2].parse::<u32>().unwrap();
    let numtrials: u32 = args[3].parse::<u32>().unwrap();
    let dimension: u32 = args[4].parse::<u32>().unwrap();

    println!("{numpoints} {numtrials} {dimension}");

    let mut avgweight = 0.0;
    for _ in 0..numtrials {
        let g = generate_graph(dimension, numpoints);
        avgweight += mst_prim(g);
    }
    avgweight = avgweight / numtrials as f64;
    println!("{avgweight} {numpoints} {numtrials} {dimension}");
    println!("Elapsed: {:.2?}", now.elapsed());
}
