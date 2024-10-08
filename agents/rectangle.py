import numpy as np
from agents.generic import GenericAgent
from agents.spiral import SpiralAgent
from utils.grid import get_center_of_tile, is_in_same_cell, get_step_size
from utils.node import Node, _make_dict_from_nparray
from utils.history import ParticleManager



class RectangleAgent(GenericAgent):
    def __init__(self, initial_position, grid, history, **kwargs):
        super(RectangleAgent, self).__init__(initial_position, grid, history, **kwargs)

        self.spID_order = np.array(-1.)
        self.step_size = tuple()
        self._state = ''  # either 'transit' or 'rectangle'
        self.position = np.array(initial_position)

        # see utils/grid.py: array of shape (n, 2) with all the left lower corners of each grid tile.
        self.grid = np.array(-1.)

        self.step_counter = 0
        self.position_history = []
        self.rectangle_containment_percentage = kwargs.get('rectangle_containment_percentage')


    def initialize(self, history, grid):
        super(RectangleAgent, self).initialize(history, grid)

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
        
        # account for second rectangle corner:
        self.spID_order = np.zeros((len(spID_order), 2+len(spID_order[0])), dtype=np.float32)
        self.spID_order[:, :-2] = np.array(spID_order)
        self._state = 'transit'
        self._update_rectangles(0)


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


    def _do_rectangle_action(self, rec1, rec2, step_size=1):
        # mode: slowly working in left-to-right or right-to-left fashion by flying vertical lines from top to bottom or vice versa.
        """
        args:
        - rec1 is the starting corner.
        - rec2 is the goal corner.
        """
        _eq = lambda x,y: abs(x-y) < self.step_size[0]


        DOWN = np.array([-step_size, 0])
        UP = np.array([step_size, 0])
        LEFT = np.array([0, -step_size])
        RIGHT = np.array([0, step_size])

        upper, lower = max(rec1[0], rec2[0]), min(rec1[0], rec2[0])

        # either 'left-to-right' or 'right-to-left'. (# TODO: rename to _horizontal_movement)
        _rectangle_state = 'left-to-right' if rec1[1] < rec2[1] else 'right-to-left'
        
        # either 'top-to-bottom' or 'bottom-to-top' -- applied at every even vertical line segment.
        _even_movement = 'bottom-to-top' if rec1[0] < rec2[0] else 'top-to-bottom'

        _vertical_parity = bool(np.round((abs(self.position[1] - rec1[1]) / self.step_size[1])) % 2)
        
        even = (_even_movement == 'top-to-bottom') and not _vertical_parity
        odd = (_even_movement == 'bottom-to-top') and _vertical_parity
        # 'DOWN' or 'UP'
        _vertical_movement = 'DOWN' if even or odd else 'UP'
        
        #print(f'state: {_rectangle_state}\neven movement: {_even_movement}\nvertical parity: {_vertical_parity}\n\n')
        
        if _eq(self.position[0], rec2[0]) and _eq(self.position[1], rec2[1]):
            self._state = 'transit'
            print(f'switching to {self._state} at time unknown because (apparently) we are done at position{self.position}')
            return self._update_position(UP)

        if _eq(self.position[0], upper) and _vertical_movement == 'UP':
            mv = RIGHT if _rectangle_state == 'left-to-right' else LEFT
            return self._update_position(mv)
        elif _eq(self.position[0], lower) and _vertical_movement == 'DOWN':
            mv = RIGHT if _rectangle_state == 'left-to-right' else LEFT
            return self._update_position(mv)

        mv = DOWN if _vertical_movement == 'DOWN' else UP


        return self._update_position(mv)


    @staticmethod
    def _center_of_gravity(particles):
        """Compute the center of gravity of the particles NOT considering their status of discovery. For the rectangle version it is (assumed to be) hindering to consider the status of discovery.
        Also, in this version, the input particles are already assumed to be filtered for the spID."""
        # If no valid particles are present, raise an exception
        if particles.size == 0:
            raise ValueError("No particles to compute the center of gravity.")
        
        return np.mean(particles[:, 1:3], axis=0)
    
    
    def _get_square_around_pt(self, pt, num_tiles):
        # calcuate left upper and right lower corners of the square:
        right_lower = pt[0] + num_tiles*self.step_size[0], pt[1] - num_tiles*self.step_size[1]
        left_upper = pt[0] - num_tiles*self.step_size[0], pt[1] + num_tiles*self.step_size[1]

        return left_upper, right_lower


    def _get_rectangle_for_superparticleid(self, spID, t):
        particles = self.history.particles[t, self.history.particles[t, :, 4] == spID]
        pt = RectangleAgent._center_of_gravity(particles)

        rcp = self.rectangle_containment_percentage
        
        nt = 0
        num_contained = 0
        square = self._get_square_around_pt(pt, nt)
        while (num_contained / particles.shape[0]) < rcp:
            nt += 1
            square = self._get_square_around_pt(pt, nt)
            num_contained = np.sum(ParticleManager.get_containment(particles, square, _check_observed=False))

        return square


    def _update_rectangles(self, t):
        for k, spID in enumerate(self.spID_order[:, 0]):
            r = self._get_rectangle_for_superparticleid(spID, t)

            # works that way, for whatever reason:
            if SpiralAgent._SpiralAgent__manhattan_distance(self.position, np.array(r[1])[None, :]) < SpiralAgent._SpiralAgent__manhattan_distance(self.position, np.array(r[0])[None, :]):
                r = r[1], r[0]

            r = [np.array(_r) for _r in r]
            r = [get_center_of_tile(_r, self.grid) for _r in r]
            self.spID_order[k, 1:3], self.spID_order[k, 5:] = r[0], r[1]


    def step(self, observation, t):
        # since we are working with axis-aligned regular grids exclusively at the moment, this method merely rounds the desired flight direction to N E S or W (in the case of self._state = transit)
        self.step_counter += 1

        # delete the corresponding super particle from the order, if we saw one.
        for i in self.spID_order:
            if np.any((observation[:, 0] == observation[:, 4]) & (observation[:, 4] == i[0])):
                self.spID_order = self.spID_order[~(self.spID_order[:, 0] == i[0])]
                self._state = 'transit'
                print(f'switching to {self._state} at time {t} because we saw something at position{self.position}')

        #print(f't={t}, state={self._state}')

        # everything discovered?
        if not len(self.spID_order):
            return self._make_node_from_position()

        if self._state == 'transit':
            # first of all, update rectangles containing the super particles
            self._update_rectangles(t)

            assert len(self.spID_order)
            
            if is_in_same_cell(self.grid, self.position, self.spID_order[0][1:3]):
                self._state = 'rectangle'
                print(f'switching to {self._state} at time {t} because we are at a rectangle')
                
                # TODO:
                # some random direction to start the rectangle: (DON'T CHOOSE (1,0)!)
                return self._update_position((0, 1))

            direction = self.spID_order[0][1:3] - self.position

            # round to axis-aligned directions
            idx = np.argmax(np.abs(direction))
            r = np.zeros_like(self.position)
            r[idx] = 1
            r *= np.sign(direction[idx])

            r = self._update_position(r)

            return r


        # in the following: self._state = rectangle
        #if not len(self.spID_order) or t >= 200:
        if not len(self.spID_order):
            return self.position

        sp_rec1, sp_rec2 = self.spID_order[0][1:3], self.spID_order[0][5:]
        
        r = self._do_rectangle_action(sp_rec1, sp_rec2)
        #try:
        #    r = self._do_rectangle_action(sp_rec1, sp_rec2)
        #except ValueError as e:
        #    print(e)
        #    return self.position
        
        return r



if __name__ == '__main__':
    pass


