import numpy as np
from agents.bnb import BranchAndBoundAgent
from agents.generic import GenericAgent
from agents.spiral import SpiralAgent
from agents.rectangle import RectangleAgent
from utils.grid import get_center_of_tile, is_in_same_cell, get_step_size
from utils.node import Node
from utils.history import ParticleManager



class RecBNBAgent(GenericAgent):
    def __init__(self, initial_position, grid, history, **kwargs):
        super(RecBNBAgent, self).__init__(initial_position, grid, history, **kwargs)

        self.spID_order = np.array(-1.)
        self.step_size = tuple()
        self._state = ''  # either 'transit' or 'rectangle'
        self.position = np.array(initial_position)

        # see utils/grid.py: array of shape (n, 2) with all the left lower corners of each grid tile.
        self.grid = np.array(-1.)

        self.step_counter = 0
        self.position_history = []
        self.rectangle_containment_percentage = kwargs.get('rectangle_containment_percentage')

        self.heuristic = kwargs.get('heuristic')
        self.static_graph_depth = kwargs.get('static_graph_depth')

        self.rectangle_paths = dict()

        self.local_uav_time_steps = 250
        if kwargs.get('local_uav_time_steps'):
            self.local_uav_time_steps = kwargs.get('local_uav_time_steps')


    def initialize(self, history, grid):
        super(RecBNBAgent, self).initialize(history, grid)

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
                                         grid=self.grid,
                                         static_graph_depth=self.static_graph_depth)
        else:
            p = self.position_history[-1]
            n = Node(particles=p.particles,
                     parent=p,
                     position=self.position)

        return n


    # small helper function. direction is in (+-1, 0), (0, +-1).
    def _update_position(self, direction):
        try:
            assert (direction[0] or direction[1]) and not (direction[0] and direction[1])
            if direction[0]:
                direction = np.sign(direction[0]) * np.array([self.step_size[0], 0.])
            else:
                direction = np.sign(direction[1]) * np.array([0., self.step_size[1]])
        except AssertionError:
            print(f'direction weird in _update_position: it is {direction}')
            direction = direction * self.step_size


        self.position += direction
        self.position = get_center_of_tile(self.position, self.grid)
        # TODO: debugging stuff:
        #print(direction)
        #assert np.all(get_center_of_tile(self.position, self.grid) == self.position)


        #self.position_history.append(self.position.copy())
        n = self._make_node_from_position()
        self.position_history.append(n)
        
        return n


    #def _do_recbnb_action(self, rec1, rec2, time):
    def _do_recbnb_action(self, _spID, time):
        """
        args:
        - rec1 is the starting corner.
        - rec2 is the goal corner.
        """
        
        #_key = (tuple(rec1), tuple(rec2))
        _key = (_spID)
        start_time, rectangle_path = self.rectangle_paths[_key]

        list_idx = time-1 - start_time
        if list_idx+1 >= len(rectangle_path):
            self._update_state('transit', time)
        
        if list_idx < len(rectangle_path):
            step_node = rectangle_path[list_idx]
        else:
            self.spID_order = self.spID_order[~(self.spID_order[:, 0] == _spID)]
            self._update_state('transit', time)
            print(f'switching to {self._state} at time {time} because BNB is done for SP {_spID}.')
            return self._update_position(np.array([1., 0.]))


        direction = (step_node.position - self.position) / self.step_size[0]
        
        # account for rounding errors:
        direction[abs(direction) < 1e-8] = 0.

        return self._update_position(direction)


    def _prepare_rectangle_action(self, rec1, rec2, _spID, start_time):
        if self.rectangle_paths.get(_spID):
            print(f'rectangle for SP {_spID} already available. continuing ...')
            return
        y_min, y_max = np.min([rec1[0], rec2[0]]), np.max([rec1[0], rec2[0]])
        x_min, x_max = np.min([rec1[1], rec2[1]]), np.max([rec1[1], rec2[1]])


        x_min -= 5*self.step_size[0]
        y_min -= 5*self.step_size[1]
        x_max += 5*self.step_size[0]
        y_max += 5*self.step_size[1]


        y_cond = (y_min < self.grid[:, 0]) & (self.grid[:, 0] < (y_max + self.step_size[0]))
        x_cond = (x_min < self.grid[:, 1]) & (self.grid[:, 1] < (x_max + self.step_size[0]))

        new_grid = self.grid[y_cond & x_cond]
        
        
        # TODO:
        local_uav_time_steps = min(self.local_uav_time_steps, self.max_uav_step - start_time-1)

        # some random values, as particles are already initialized and adjusted from before and will just be set.
        new_history = ParticleManager(1, local_uav_time_steps, None, self.history.file_config)
        new_history.particles = self.history.particles[start_time:start_time+local_uav_time_steps].copy()


        local_agent = BranchAndBoundAgent(initial_position=self.position,
                                          grid=new_grid,
                                          uav_time_steps=local_uav_time_steps,
                                          history=new_history,
                                          heuristic=self.heuristic)

        
        _, best_path = local_agent.solve()

        #_key = (tuple(rec1), tuple(rec2))
        _key = _spID
        self.rectangle_paths[_key] = (start_time, best_path)

        new_particles = best_path[-1].particles
        self.history.particles[(start_time + local_uav_time_steps):, :, 3] = new_particles[local_uav_time_steps-1][:, 3]

        print(f'{int(np.sum(new_particles[local_uav_time_steps-1][:, 3]))} particles were discovered for this rectangle from time {start_time} to {start_time + local_uav_time_steps}')



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
        while ((num_contained / particles.shape[0]) < rcp) and nt < 25:
            nt += 1
            square = self._get_square_around_pt(pt, nt)
            num_contained = np.sum(ParticleManager.get_containment(particles, square, _check_observed=False))

        #print(f'final nt: {nt}')

        return [get_center_of_tile(np.array(s), self.grid) for s in square]


    def _update_rectangles(self, t):
        for k, spID in enumerate(self.spID_order[:, 0]):
            # do not update particles, that were already discovered:
            if self.history.particles[t, int(spID)-1, 3]:
                continue

            r = self._get_rectangle_for_superparticleid(spID, t)

            # works that way, for whatever reason:
            if SpiralAgent._SpiralAgent__manhattan_distance(self.position, np.array(r[1])[None, :]) < SpiralAgent._SpiralAgent__manhattan_distance(self.position, np.array(r[0])[None, :]):
                r = r[1], r[0]

            r = [np.array(_r) for _r in r]
            r = [get_center_of_tile(_r, self.grid) for _r in r]
            self.spID_order[k, 1:3], self.spID_order[k, 5:] = r[0], r[1]


    def _update_state(self, new_state, t):
        if new_state == 'rectangle':
            #self._prepare_rectangle_action(rec1=self.spID_order[0][1:3],
            #                               rec2=self.spID_order[0][5:],
            #                               start_time=t)
            self._prepare_rectangle_action(rec1=self.spID_order[0][1:3],
                                           rec2=self.spID_order[0][5:],
                                           _spID=self.spID_order[0][0],
                                           start_time=t)
            self._state = new_state
        elif new_state == 'transit':
            self._state = new_state


    def step(self, observation, t):
        # since we are working with axis-aligned regular grids exclusively at the moment, this method merely rounds the desired flight direction to N E S or W (in the case of self._state = transit)
        self.step_counter += 1

        # delete the corresponding super particle from the order, if we saw one.
        for i in self.spID_order:
            if np.any((observation[:, 0] == observation[:, 4]) & (observation[:, 4] == i[0])):
                self.spID_order = self.spID_order[~(self.spID_order[:, 0] == i[0])]
                self._update_state('transit', t)
                #print('notnotnot')
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
                self._update_state('rectangle', t)
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

        sp_rec1, sp_rec2, _spID = self.spID_order[0][1:3], self.spID_order[0][5:], self.spID_order[0][0]
        
        #r = self._do_recbnb_action(sp_rec1, sp_rec2, t)
        r = self._do_recbnb_action(_spID, t)
        #try:
        #    r = self._do_rectangle_action(sp_rec1, sp_rec2)
        #except ValueError as e:
        #    print(e)
        #    return self.position
        
        return r



if __name__ == '__main__':
    pass


