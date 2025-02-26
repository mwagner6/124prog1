import sys
import math
import random
import time

# We'll use the d-heap outlined in lecture 6 for Prim's
class dHeap:
    def __init__(self, d=2):
        self.d = d
        self.heap = []

    def parent(self, i):
        return (i - 1) // self.d
    
    def children(self, i):
        return range(1+(i * self.d), 1+(i * self.d)+self.d)
    
    # verifies upwards from i
    def verify_up(self, i):
        if i == 0:
            return
        p = self.parent(i)
        if self.heap[i][0] < self.heap[p][0]:
            self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
            self.verify_up(p)
    
    def verify_down(self, i):
        ci = self.children(i)
        swap = False
        min = self.heap[i][0]
        minIdx = i
        for idx in ci:
            if idx >= len(self.heap):
                break
            if self.heap[idx][0] < min:
                minIdx = idx
                min = self.heap[idx][0]
                swap = True
        if swap:
            self.heap[i], self.heap[minIdx] = self.heap[minIdx], self.heap[i]
            self.verify_down(minIdx)


    def push(self, item):
        self.heap.append(item)
        self.verify_up(len(self.heap)-1)


    def pop(self):
        if not self.heap or len(self.heap) < 1:
            raise IndexError("trying to pop from an empty heap")
        out = self.heap[0]
        if len(self.heap) > 1:
            self.heap[0] = self.heap.pop(-1)
            self.verify_down(0)
        else:
            self.heap.pop()
        return out


class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = {i: [] for i in range(n)}

    def add_edge(self, u, v, weight):
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))

    def neighbors(self, u):
        return self.adj[u]


# We'll have dense (because they're complete) graphs, so Prim's makes sense
def MST_prim(graph, d=2):
    """
    For all v ∈ V, d[v] ← ∞; S = ∅; Create(H)
    d[s] = 0; Prev[s] ← null; Insert(H, s, 0)
    While H ≠ ∅ do
    u ← deletemin(H); S ← S ∪ {u}
    For each edge (u, v) ∈ E, with v ∉ S do
    If d[v] > w((u, v)) then
    d[v] = w((u, v)); Prev[v] = u; Insert(H, v, d[v])
    Return: d[·], Prev[·]

    """
    n = graph.n
    visited = [False] * n # to not revisit (added)
    d_val = [float('inf')] * n # the smallest weight possible for an edge that connects this node v to the MST
    prev = [None] * n # what node value led to this node index


    s = 0
    d_val[s] = 0
    heap = dHeap(d)
    heap.push((0, s))

    while heap and len(heap.heap) > 0:
        key, u = heap.pop()
        if visited[u]:
            continue
        visited[u] = True
        for (v, weight) in graph.neighbors(u):
            if (not visited[v]) and d_val[v] > weight:
                d_val[v] = weight
                prev[v] = u
                heap.push((weight, v))

    total_weight = sum(d_val)
    return total_weight # make it return total_weight as well for the progset purposes


def complete_basic(n):
    g = Graph(n)
    for u in range(n-1):
        for v in range(u+1, n):
            g.add_edge(u, v, random.random())
    return g

def hypercube(n):
    pass

def complete_unit_square(n):
    pass

def complete_unit_cube(n):
    pass

def complete_hypercube(n):
    pass


def main():
    start = time.time()
    arg0 = int(sys.argv[1])
    numpoints = int(sys.argv[2])
    numtrials = int(sys.argv[3])
    dimension = int(sys.argv[4])

    # Debug: Print the arguments to confirm they were captured correctly.
    print("Argument 0:", arg0)
    print("Number of points:", numpoints)
    print("Number of trials:", numtrials)
    print("Dimension:", dimension)

    func = {0: complete_basic, 1: hypercube, 2: complete_unit_square, 3: complete_unit_cube, 4: complete_hypercube}
    avgweight = 0
    for i in range(numtrials):
        g = func[dimension](numpoints)
        avgweight += MST_prim(g)
    avgweight /= numtrials
    print(f"{avgweight} {numpoints} {numtrials} {dimension}")
    print(f"Time elapsed: {time.time() - start}")

if __name__ == "__main__":
    main()