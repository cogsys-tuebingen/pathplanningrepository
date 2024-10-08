from collections import defaultdict
import numpy as np
from utils.grid import get_contours_of_tile, get_neighbouring_points
from utils.history import ParticleManager
from utils.random_crap import get_simulation_time
from functools import cache

# V represents the number of Vertices, present 
# in the given DAG.
V=6
# INF means infinity, which is taken
# as a very large number.
# DummyNode class
class DummyNode:
    # v is the vertex, 
    # and wt is the weight.
    def __init__(self, _v, _wt):
        self.v = _v
        self.wt = _wt

        self.next_for_shortest = None


    def __repr__(self) -> str:
        return str(self.v)


class StaticNode:
    """
    Class to represent the nodes with space (lat, lon) and time coordinates in the grid to search with the UAV.
    particles: pointer to particles array (T, num_particles, 5).
    """

    def __init__(self, particles, position, grid):
        self.particles = particles

        self._visited = defaultdict(lambda: False)
        self.pos = position
        self.grid = grid
    

    def visited(self, t):
        return self._visited[t]


    def generate_children(self):
        return get_neighbouring_points(self.pos, self.grid)

    def negative_probability_of_detection(self, time:int, normalize_over_targets=True):
        # TODO: substitute explicit floats by Arguments
        sim_time = get_simulation_time(time, 1, 0.1)

        prob = self.observe(sim_time)
        div = len(prob) if normalize_over_targets else 1

        return np.sum(prob[:, 1] / prob[:, 2]) / div

    @cache
    def observe(self, time):
        # actual observation and update of self.particles:
        tile = get_contours_of_tile(self.pos, self.grid)
        obs = ParticleManager.get_observation(self.particles, tile, time)

        now_particles = self.particles[time]

        # TODO: does this work correctly:?
        sp = ParticleManager.get_super_particles(self.particles)
        particle_info = np.zeros((len(sp), 3))
        for k, spid in enumerate(np.arange(1, 1+len(particle_info))):
            particle_info[k, 0] = spid
            particle_info[k, 1] = np.sum(obs[:, 4] == spid)
            particle_info[k, 2] = np.sum(now_particles[:, 4] == spid)

        return particle_info


# Graph Class
class Graph:
    """
    self.vertices: defaultdict carrying all the nodes that were visited already so they only need to be constructed once visited.
    """
    def __init__(self, max_time):
        # Initializing Adajency list.
        self.adj = defaultdict(list)
        self.adj['len'] = 0

        self.vertices = defaultdict(lambda: None)

        self.max_time = max_time

        self.INF = -999999999999


    # Function to find topological Sort which is always possible
    # as the given graph is a DAG.
    # NOTE: this does not perform a topological sort as not all nodes are included. the nodes, which are unreachable from the root node, are not included. this is fine for the application of finding a shortest path from a given root node.
    # TODO: switch to non-recursive implementation for efficiency
    def topologicalSort(self, v, st, time):
        # Marking v as visited.
        
        v._visited[time] = True

        # TODO: Abbruchkriterium


        # Iterating for all the adjacent nodes of v. (if time is not over)
        if time < self.max_time:
            for coordinates in v.generate_children():
                coord_idx = tuple(coordinates)

                if self.vertices[coord_idx]:
                    node = self.vertices[coord_idx]
                else:
                    node = StaticNode(v.particles, coordinates, v.grid)
                    self.vertices[coord_idx] = node
                
                # If any adjacent node to v is not 
                # visited, call topologicalSort function on it.
                if(node.visited(1+time) == False):
                    self.topologicalSort(node, st, 1+time)


        # Push v into Stack
        idx = st['len']
        st[idx] = (v, time)
        st['len'] += 1



    #def numpy_topological_sort(self, s, adj_matrix, label_dict):
    #    sorted_spacers = []
    #    while s:
    #        n = s.pop()
    #        sorted_spacers.append(n)
    #        for j in range(adj_matrix.shape[1]):
    #            if adj_matrix[n, j] == 1:
    #                adj_matrix[n, j] = 0
    #                if sum(adj_matrix[:, j]) == 0:
    #                    s += [j]
    #    if np.all(adj_matrix == 0):
    #        return [label_dict[i] for i in sorted_spacers]
    #    else:
    #        raise Exception(f'Spacer arrays contain (at least one) cycle')

    

    # Function to compute the shortest path 
    # to all other vertices starting from src.
    def shortest_path(self, src:StaticNode, start_time:int):
        
        # Declare a Stack (st) which is used to find 
        # the topological sort of the given DAG.
        st=[]
        st = {'len': 0}


        t = start_time

        # Declare a dist array where dist[i] denotes
        # shortest distance of src from i. 
        # Initialize all it's entries with self.INF and 
        # dist[src] with 0.

        # TODO: adjust this for vertices to be something else than consecutive numbers:
        dist = {k: self.INF for k in self.adj}
        dist = defaultdict(lambda : self.INF)

        dist[src] = 0

        # Create boolean visited array to keep track 
        # of visited elements.
        #visited = defaultdict(lambda: False)
        # replaced by StaticNode.visited[time]

        # Iterate for all the V vertices.
        for c in src.generate_children():
            c_idx = tuple(c)
            if not self.vertices[c_idx]:
                self.vertices[c_idx] = StaticNode(src.particles, c, src.grid)
                cn = self.vertices[c_idx]

            # If 'i' found to unvisited call 
            # topoplogicalSort from there.
            if cn.visited(1+t) == False:
                # TODO: change function call (after topologicalSort is changed)
                self.topologicalSort(cn, st, 1+t)
            
        
        
        # Iterate till the stack is not empty.
        # > 1 because we have to make up for the 'len' entry.
        while len(st) > 1:
            # Pop element from stack and store it in u.
            st['len'] -= 1
            idx = st['len']
            u, t = st.pop(idx)

            # If shortest distance from src to u is 
            # not infinity i.e. it is reachable.

            if dist[u] != self.INF:
                # Iterate for all the adjacent vertices 
                # of u.
                for coord in u.generate_children():
                    coord_idx = tuple(coord)
                    if not self.vertices[coord_idx]:
                        self.vertices[coord_idx] = StaticNode(u.particles, coord, u.grid)
                    node = self.vertices[coord_idx]

                    # If distance of src->v is greater than
                    # distance of src->u + u->v then update
                    # the value as shown.
                    neg_pod = node.negative_probability_of_detection(t-1)
                    if dist[node] > dist[u] + neg_pod:
                        dist[node] = dist[u] + neg_pod

        
        # print the distances.
        #for i in self.adj:
        #    if dist[i]==self.INF:
        #        print("self.INF"),
        #    else:
        #        print(f'distance to {i}: {dist[i]}'),

        return dist


if __name__ == '__main__':
    g=Graph(V)
    # Add edges.
    #g.addEdge(0, 2, 4)
    #g.addEdge(1, 0, 3)
    #g.addEdge(2, 3,-3)
    #g.addEdge(2, 4, 2)
    #g.addEdge(1, 2, 2)
    #g.addEdge(1, 3, 5)
    #g.addEdge(4, 3, 2)


    g.addEdge(0, 1, 4)
    g.addEdge(1, 2, 3)
    g.addEdge(3, 4, 1)
    g.addEdge(4, 2, 2)
    g.addEdge(2, 5, 2)
    g.addEdge(0, 3, 2)

    stack = {'len': 0}
    visited = defaultdict(lambda: False)
    g.topologicalSort(0, visited, stack)
    print(list(stack.values())[1:])


    # Find the shortest path from a 
    # vertex (here 0).
    g.shortest_path(0)

    #wobei s die start enden sind:

    ## find spacers/nodes with no incoming edges (roots)
    #s = []
    #for j in range(adj_matrix.shape[1]):
    #    if all(a == 0 for a in adj_matrix[:, j]):
    #        s += [j]


