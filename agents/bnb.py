import time

from agents.generic import GenericAgent
from utils.node import Node
from utils.grid import get_center_of_tile

from queue import PriorityQueue
import numpy as np


class BranchAndBoundAgent(GenericAgent):
    def __init__(self, initial_position:np.ndarray, grid:np.ndarray, history, **kwargs):
        super(BranchAndBoundAgent, self).__init__(initial_position, grid, history, **kwargs)

        self.heuristic = kwargs.get('heuristic')
        self.static_graph_depth = kwargs.get('static_graph_depth')
        self.vicinity_weight = kwargs.get('vicinity_weight')

    
    def initialize(self, history, grid):
        super(BranchAndBoundAgent, self).initialize(history, grid)
        
        s = time.time()
        r = self.solve()
        print(f'it took {time.time() - s} seconds to solve ...')
        return r


    @staticmethod
    def _get_path_from_final_node(final_node):
        rev_path = [final_node]
        p_node = final_node.parent
        while p_node.parent is not None:
            rev_path.append(p_node)
            p_node = p_node.parent

        return rev_path[::-1]


    def solve(self):
        # from 2008er Diss p 15
    
        # Step 0:
        t = 1

        # Initialize a queue with a root node
        # of the branch and bound tree
        qs = {k: PriorityQueue() for k in range(1+self.max_uav_step)}

        s = Node.generate_start_node(particles=self.history.particles,
                                     max_uav_step=self.max_uav_step,
                                     position=self.initial_position,
                                     grid=self.grid,
                                     heuristic=self.heuristic,
                                     static_graph_depth=self.static_graph_depth,
                                     vicinity_weight=self.vicinity_weight)

        for n in s.generate_children(self.grid):
            # negative values, because PriorityQueue always returns the lowest value
            #qs[t].put((-n.value(), n))
            qs[t].put(n)

        # Initialize a variable to keep track
        # of the maximum probability found so far
        # (equals \hat{q} from Algo in 2008er Diss)
        max_prob = -1.0
        # best_final_node is there to keep track of the incumbent path.
        best_final_node = s

        c = 0
        # [Control Flow Steps] 1, 2, and 3:
        #while t > 0:
        start = time.time()
        while t > 0 and c <= 200000:
            c += 1
            if not (c % 10000):
                print(f'number of iterations: {c}')


            if not qs[t].empty():
                # Step 3:
                n = qs[t].get()
            else:
                # Step 2:
                t -= 1
                continue
            #print(f'looking at {n}')

            # Step 4:
            if n.value() <= max_prob:
                continue

            # Step 5:
            if t >= -1 + self.max_uav_step:
                # (here, we already know that max_prob < n.value by step 4; hence the update)
                max_prob = n.value()
                # save incumbent path:
                best_final_node = n
                #print(f'node update: {n}, max_prob = {max_prob}')


                #TODO
                #break

            else:
                t += 1
                for v in n.generate_children(self.grid):
                    qs[t].put(v)


        print(f'it took {c} iterations to finish')
        
        end = time.time()
        print(f'it took {end-start} seconds to calculate this trajectory ...')
        self.best_path = self._get_path_from_final_node(best_final_node)
        return max_prob, self.best_path
    
    
    def step(self, observation, idx):
        #r = self.best_path[idx].position
        # trying to switch to nodes instead of np arrays now.
        r = self.best_path[idx]
        self.position_history.append(r)

        return r


# Driver Code
if __name__ == '__main__':
    W = 10
    arr = [Item(2, 40, 'a'), Item(3.14, 50, 'b'), Item(
        1.98, 100, 'c'), Item(5, 95, 'd'), Item(3, 30, 'e')]

    print(arr)

    bnb = BranchAndBoundAgent(max_weight=W)

    max_profit, best_final_node = bnb.solve(arr)

    full_path = [best_final_node]
    
    while full_path[-1].parent:
        full_path.append(full_path[-1].parent)
        

    print('Maximum possible profit =', max_profit, f'\nwith path: {full_path[::-1][1:]}')


