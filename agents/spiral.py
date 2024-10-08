import numpy as np
from agents.generic import GenericAgent
from utils.grid import get_center_of_tile, is_in_same_cell, get_step_size
from utils.node import Node, _make_dict_from_nparray



class SpiralAgent(GenericAgent):
    def __init__(self, initial_position, grid, history, **kwargs):
        super(SpiralAgent, self).__init__(initial_position, grid, history, **kwargs)

        self.spID_order = np.array(-1.)
        self.step_size = tuple()
        self._state = ''  # either 'transit' or 'spiral'
        self._spiral_state = 'north'  # compass headings; defaults to north 'cause Jon Snow
        self.position = np.array(initial_position)

        # see utils/grid.py: array of shape (n, 2) with all the left lower corners of each grid tile.
        self.grid = np.array(-1.)

        self.step_counter = 0
        self.position_history = []

        self.timing_history = []
    
    
    @staticmethod
    def __manhattan_distance(pos, pos_vec):
        """
        pos is of shape (2,), pos_vec is of sape (n, 2).
            returns: array of shape (n, ) with manhattan distance from pos to pos_vec.
        """
        abs_dist = np.abs(pos - pos_vec)

        return abs_dist[:, 0] + abs_dist[:, 1]

    @staticmethod
    def _find_closest(pos, particles):
        return np.argmin(SpiralAgent.__manhattan_distance(pos, particles[:, 1:3]))

    def initialize(self, history, grid):
        super(SpiralAgent, self).initialize(history, grid)

        self.step_size = get_step_size(self.grid)
        sp = history._get_super_particles()
        pos = self.position
        spID_order = list()
        for _ in range(len(sp)):
            closest_idx = SpiralAgent._find_closest(pos, sp)
            #spID_order.append(sp[closest_idx])
            s = sp[closest_idx].copy()
            s[1:3] = get_center_of_tile(s[1:3], self.grid)
            spID_order.append(s)
            sp = np.delete(sp, closest_idx, axis=0)

        self.spID_order = np.array(spID_order)
        self._state = 'transit'
        self._update_assumed_superparticle_locations(0)


    def _make_node_from_position(self):
        if not self.position_history:
            n = Node.generate_start_node(particles=self.history.particles,
                                         max_uav_step=self.max_uav_step,
                                         position=self.position,
                                         grid=self.grid)
        else:
            p = self.position_history[-1]
            n = Node(particles=p.particles,
                     parent=p,
                     position=self.position)

        return n


    # small helper function. direction is in (+-1, 0), (0, +-1).
    def _update_position(self, direction):
        assert (direction[0] or direction[1]) and not (direction[0] and direction[1])
        if direction[0]:
            direction = np.sign(direction[0]) * np.array([self.step_size[0], 0.])
        else:
            direction = np.sign(direction[1]) * np.array([0., self.step_size[1]])

        self.position += direction
        #self.position_history.append(self.position.copy())
        n = self._make_node_from_position()
        self.position_history.append(n)
        
        return n
    
    # etwas random, aber die Richtungen sind transponiert:
    _p = {
            'east': (0, 1),
            'west': (0, -1),
            'north': (1, 0),
            'south': (-1, 0)
         }
    # helper for the spiral direction finding logic.
    def _do_spiral_action(self, sp_coord):
        x, y = self.position - sp_coord
        x, y = np.round(x / self.step_size[0]), np.round(y / self.step_size[1])

        #print('rel. pos: ', x, y, end='\n')

        #print(f'(x, y)=({x}, {y})')
        #print(f'moving {self._spiral_state}')

        if self._spiral_state == 'north':
            if y == -x:
                self._spiral_state = 'east'
                return self._do_spiral_action(sp_coord)

        if self._spiral_state == 'south':
            if x == -y:
                self._spiral_state = 'west'
                return self._do_spiral_action(sp_coord)

        if self._spiral_state == 'west':
            if -x == -y:
                self._spiral_state = 'north'
                return self._do_spiral_action(sp_coord)


        # special case east! (extending spiral)
        if self._spiral_state == 'east':
            # also looks weird, one would expect condition to be x == 1+y; but has to be done like this because of transposition:
            if 1+x == y:
                self._spiral_state = 'south'
                return self._do_spiral_action(sp_coord)

        return self._update_position(SpiralAgent._p[self._spiral_state])


    @staticmethod
    def _center_of_gravity(particles, spID):
        """Compute the center of gravity of the particles considering their status of discovery."""
        
        if particles.shape[1] != 5:
            raise ValueError("Input must be a (n, 5)-shaped numpy array.")
        
        # Filter vectors where the fourth column is 0
        valid_particles = particles[particles[:, 3] == 0]
        # Filter vectors for given super particle ID
        valid_particles = valid_particles[valid_particles[:, 4] == spID][:, 1:3]
        
        # If no valid particles are present, raise an exception
        if valid_particles.size == 0:
            raise ValueError("No valid particles to compute the center of gravity.")
        
        return np.mean(valid_particles, axis=0)


    def _update_assumed_superparticle_locations(self, t):
        particles = self.history.particles[t]

        for k, spID in enumerate(self.spID_order[:, 0]):
            cog = SpiralAgent._center_of_gravity(particles, spID)
            self.spID_order[k, 1:3] = get_center_of_tile(cog, self.grid)



    def step(self, observation, t):
        # since we are working with axis-aligned regular grids exclusively at the moment, this method merely rounds the desired flight direction to N E S or W (in the case of self._state = transit)
        self.step_counter += 1

        # delete the corresponding super particle from the order, if we saw one.
        for i in self.spID_order:
            if np.any((observation[:, 0] == observation[:, 4]) & (observation[:, 4] == i[0])):
                self.spID_order = self.spID_order[~(self.spID_order[:, 0] == i[0])]
                self._state = 'transit'

        # everything discovered?
        if not len(self.spID_order):
            return self._make_node_from_position()

        if self._state == 'transit':
            # first of all, update assumed position of super particles
            self._update_assumed_superparticle_locations(t)

            assert len(self.spID_order)
            
            if is_in_same_cell(self.grid, self.position, self.spID_order[0][1:3]):
                self._state = 'spiral'
                
                # TODO:
                self._spiral_state = 'south'
                # some random direction to start the spiral: (DON'T CHOOSE (1,0)!)
                return self._update_position((0, 1))

            direction = self.spID_order[0][1:3] - self.position

            # round to axis-aligned directions
            idx = np.argmax(np.abs(direction))
            r = np.zeros_like(self.position)
            r[idx] = 1
            r *= np.sign(direction[idx])

            r = self._update_position(r)

            return r

        
        # in the following: self._state = spiral
        if not len(self.spID_order):
            return self.position

        sp_cell_midpoint = self.spID_order[0][1:3]

        r = self._do_spiral_action(sp_cell_midpoint)
        
        return r



if __name__ == '__main__':
    pass


