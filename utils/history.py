from utils.grid import get_center_of_tile
from utils.random_crap import get_simulation_time, meter_distance_from_latlon, get_adjusted_uav_time_steps

from sklearn.neighbors import NearestNeighbors
import numpy as np


class ParticleManager:
    # class to manage (and provide appropriate getters for) the history provided by a run of a leeway simulation.
    def __init__(self, num_particles:int, num_timesteps:int=None, run_history:np.ndarray=None, file_config:dict=None):
        self.data = run_history
        
        # this looks like: shape = (T, n, 5), where n is the number of particles, T the number of time steps.
        # one row looks like ID, lat, lon, discovered, id of associated super particle
        self.particles = -1 * np.ones(shape=(1, num_particles, 5), dtype=np.float32)
        self.observation_history = []

        self.original_particles = self.particles.copy()
        self.max_uav_step = -1

        self.file_config = file_config




    @staticmethod
    def _k_nearest_neighbours(particles, time, position, k):
        # Create a numpy array of coordinates
        #coordinates = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        coordinates = particles[time, :, 1:3]

        # Create a NearestNeighbors model
        knn_model = NearestNeighbors(n_neighbors=k)
        knn_model.fit(coordinates)

        # Find the k-nearest neighbors and their distances
        distances, indices = knn_model.kneighbors([position])

        # Print the indices of k-nearest neighbors and their distances
        #print("Indices of nearest neighbors:", indices)
        #print("Distances to nearest neighbors:", distances)

        points = particles[time, indices, 1:]
        
        return points, distances, indices


    def _get_observation(self, tile:tuple, t:int, grid:np.ndarray):
        """
        Args:
            tile: tuple of two np.arrays of shape (2,); left_upper and right_lower of an axis-aligned tile of the grid. (like return of utils.grid.get_contours_of_tile)
            grid: grid on which we are working (like from utils.grid)
        returns: subarray of self.particles which is contained in the observed grid cell.
        """
        #if 7.17291281 < tile[0][1] < 7.17381274:
        #    print('jojo from history')


        # observation_history only contains the center point of the observed tile
        # get_center_of_tile apparently only returns the center of the correct tile, when using tile[1], not tile[0].
        center_tile = get_center_of_tile(tile[1], grid)
        self.observation_history.append(center_tile)

        obs, _ = ParticleManager.get_observation(self.particles[t], tile)

        self.particles[np.isnan(self.particles)] = 500

        #s = ''
        #s = f'\n\nposition = {center_tile}\nnearest:\n{ParticleManager._k_nearest_neighbours(self.particles, t, center_tile, 5)[:2]}'
        #if len(obs):
        #    idxs = ParticleManager._k_nearest_neighbours(self.particles, t, center_tile, 5)[2].squeeze()
        #    p = self.particles[t, idxs]
        #    p[:, 0] = meter_distance_from_latlon(p[:, 1:3], center_tile)
        #    s = f'\n\nposition = {center_tile} , time = {t}\nnearest:'
        #    s += f'\n(corrected) actual observation:\n{p}'
        #    print(s)


        # account for the fact, that get_observation now only marks things as observed at t, not at t, t+1, t+2, ....
        if t < (len(self.particles) -1):
            self.particles[(t+1):, :, 3] = self.particles[t, :, 3]
        

        return obs


    @staticmethod
    def get_containment(time_state:np.ndarray, tile:tuple, _check_observed=True):
        # check vs left upper:
        idx_vec = (time_state[:, 1] > tile[0][0]) & (time_state[:, 2] < tile[0][1])
        # check vs right lower:
        idx_vec &= (time_state[:, 1] < tile[1][0]) & (time_state[:, 2] > tile[1][1])
        # check that it was not observed already:

        test_number = np.sum(idx_vec)

        if _check_observed:
            idx_vec &= ~time_state[:, 3].astype(np.bool_)

        if test_number != np.sum(idx_vec):
            pass
            #print('break')

        return idx_vec


    @staticmethod
    def get_observation(time_state:np.ndarray, tile:tuple, copy_if_observed:bool=False):
        """
        Args:
            time_state: np.array of shape (num_particles, 5); basically, this is a reference to particles[t] for some t
            tile: tuple of two np.arrays of shape (2,); left_upper and right_lower of an axis-aligned tile of the grid. (like return of utils.grid.get_contours_of_tile)
        returns: obs: subarray of particles which is contained in the observed grid cell.
                 time_state: if copy_if_observed is true, this is an updated version of particles[t].
        """
        #time_state = particles[t]

        idx_vec = ParticleManager.get_containment(time_state, tile)

        obs = time_state[idx_vec].astype(np.int32)
        
        # TODO: is this for loop equivalent to below's numpy expression?
        #for i in obs[:, 0]:
        #    # mark as discovered:
        #    particles[t:, int(i-1), 3] = 1
        
        if copy_if_observed and np.any(obs):
            time_state = time_state.copy()

        time_state[obs[:, 0]-1, 3] = 1

        return obs, time_state

    
    @staticmethod
    def get_cone_observation(time_state:np.ndarray, position:np.ndarray, compass_heading:np.ndarray, _count_observed=False):
        """
        args:
            time_state: particles; np,array of shape (n, 5)
            position: np.array of shape (2,)
            compass_heading: normalized (2,) np.array with only one non-vanishing entry.
        """
        # Define compass heading to direction mapping
        # (take into account that coordinate system is transposed)
        #headings = {
        #    'east': np.array([0, 1]),
        #    'west': np.array([0, -1]),
        #    'north': np.array([1, 0]),
        #    'south': np.array([-1, 0])
        #}

        # Get the direction vector
        #D = headings[compass_heading]
        D = compass_heading

        time_state = time_state[~time_state[:, 3].astype(bool)]

        # Calculate vectors from position to time_state
        vectors_OP = time_state[:, 1:3] - position

        # Normalize these vectors
        normalized_OP = vectors_OP / np.linalg.norm(vectors_OP, axis=1, keepdims=True)

        # Check if each particle is within the cone using dot product
        inside_cone = np.dot(normalized_OP, D) > np.cos(np.pi / 4) # cos(90/2)

        return time_state[inside_cone]


    def initialize(self, run_history):
        if not self.data:
            self.data = run_history
            self.num_timesteps = len(run_history[0])
            self.num_particles = len(run_history)

            self.particles = -1 * np.ones(shape=(self.num_timesteps, self.num_particles, 5), dtype=np.float32)
        else:
            raise NotImplementedError

    def _init_superparticle(self, id_super_particle: int, range_following_particle_ids: tuple):
        # TODO: der ganze Bums hier initialisiert alles nur zum Zeitpunkt t=0. natürlich muss in der history jeder Zeitpunkt abgebildet sein!!1!

        rfp = range_following_particle_ids
        idsp = id_super_particle
        
        for t in range(self.num_timesteps):
            # in run history, the order of lat and lon is switched
            s_lon, s_lat = self.data[idsp-1][t][5], self.data[idsp-1][t][6]

            super_particle_array = np.array([idsp, s_lat, s_lon, 0., idsp], dtype=np.float32)
            assert self.particles[t][idsp-1][0] == -1
            self.particles[t][idsp-1] = super_particle_array


            particle_array = np.zeros((1, rfp[1] + 1 - rfp[0], 5), dtype=np.float32)

            for k in range(rfp[0], 1+rfp[1]):
                idx = k - rfp[0]
                # TODO: ist das nicht falsch, immer auf self.data[0] zugreifen?
                s_lon, s_lat = self.data[k-1][t][5], self.data[k-1][t][6]
                particle_array[0, idx] = np.array([k, s_lat, s_lon, 0, idsp])
            
            assert np.sum(self.particles[t, rfp[0]-1:rfp[1], 0] != -1) == 0
            self.particles[t, rfp[0]-1:rfp[1]] = particle_array


    # big picture wise, still, this class should manage its own structure containing the particles and not make use of the run history. too complicated, too annoying.
    def add_particles(self, ids_super_particle: list, range_following_particle_ids: list):
        assert len(ids_super_particle) == len(range_following_particle_ids)
        for idsp, rfpi in zip(ids_super_particle, range_following_particle_ids):
            self._init_superparticle(idsp, rfpi)

        #self.particles[np.isnan(self.particles)] = 500
        #self.particles = self.particles[~np.isnan(self.particles)]
        nan_cols = np.any(np.isnan(self.particles), axis=(0, 2))
    
        # Use fancy indexing to remove those columns
        #cleaned_arr = self.particles[:, ~nan_cols, :]
        # TODO: leider BS
        #self.particles[:, nan_cols] = self.particles[:, 0][:, None, :]
        self.particles[:, nan_cols, 4] = -1

        if np.any(np.isnan(self.particles)):
            print(f'corrected some NaNs in particles. {np.sum(nan_cols)} in total ...')

        #self.particles = cleaned_arr

    
    def _get_super_particles(self):
        return ParticleManager.get_super_particles(self.particles)

    @staticmethod
    def get_super_particles(particles):
        """
        returns: the actual entries of the superparticle, with shape (m, 5), where m is the number of super particles. so the ID of the second superparticle could be accesses by rp[1, 0], for example
        """
        # index of particles whose associated super particles are themselves (aka the super particles)
        b = particles[:, :, 0] == particles[:, :, 4]

        rp = particles[0][b[0].nonzero()]
        #print('returning following super particles:',rp)

        return rp
        #return self.particles[b]



    def adjust_particles(self, max_uav_step, uav_step_start=0):
        self.max_uav_step = max_uav_step
        self.original_particles = self.particles.copy()
        
        s = self.original_particles.shape

        self.particles = np.zeros((self.max_uav_step, *s[1:]))


        simulation_time_step_length_min = self.file_config['settings']['simulation_interval_minutes']
        tile_size_km = self.file_config['grid_settings']['tile_size_km']

        for k in range(uav_step_start, self.max_uav_step):
            t = get_simulation_time(k, simulation_time_step_length_min, tile_size_km)

            self.particles[k] = self.original_particles[t].copy()

