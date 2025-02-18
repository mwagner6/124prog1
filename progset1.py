import sys
import math
import random

# We'll use the d-heap outlined in lecture 6 for Prim's
class dHeap:
    def __init__(self, d=2):
        self.d = d
        self.heap = []

    def parent(self, i):
        return (i - 1) // self.d
    
    def children(self, i):
        return range(self.d)+1 + (i * self.d)
    
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
            if self.heap[idx] < min:
                minIdx = idx
                min = self.heap[idx]
                swap = True
        if swap:
            self.heap[i], self.heap[minIdx] = self.heap[minIdx], self.heap[i]
            self.verify_down(minIdx)


    def push(self, item):
        self.heap.append(item)
        self.verify_up(len(self.heap)-1)


    def pop(self):
        if not self.heap:
            raise IndexError("trying to pop from an empty heap")
        out = self.heap[0].copy()
        self.heap[0] = self.heap.pop(-1)
        self.verify_down(0)
        return out


class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = {i: [] for in range(n)}

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

    while heap:
        key, u = heap.pop()
        if visited[u]:
            continue
        visited[u] = True

    for (v, weight) in graph.neigbors(u):
        if not visited[u] and d_val[v] > weight:
            d_val[v] = weight
            prev[v] = u
            heap.push((weight, v))

    total_weight = sum(d_val)
    return total_weight, d_val, prev # make it return total_weight as well for the progset purposes


    def complete_basic(n):
        pass

    def hypercube(n):
        pass

    def complete_unit_square(n):
        pass

    def complete_unit_cube(n):
        pass

    def complete_hypercube(n):
        pass


def main():
    arg0 = int(sys.argv[1])
    numpoints = int(sys.argv[2])
    numtrials = int(sys.argv[3])
    dimension = int(sys.argv[4])

    # Debug: Print the arguments to confirm they were captured correctly.
    print("Argument 0:", arg0)
    print("Number of points:", numpoints)
    print("Number of trials:", numtrials)
    print("Dimension:", dimension)

if __name__ == "__main__":
    main()