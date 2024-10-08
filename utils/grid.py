import numpy as np
from math import sqrt


def generate_grid(center, tile_size_km, grid_size_km):
    """
    Generates a grid of tiles in lat-lon coordinates around a center point.
    
    Args:
        center (tuple): Center point coordinates in (latitude, longitude) format.
        tile_size_km (float): Size of each tile in kilometers.
        grid_size_km (float): Total size of the grid in kilometers.
        
    Returns:
        np.ndarray: each row is the coordinates of the tiles in the grid (left lower corner of each tile).
    """

    # TODO: dieser bums rechnet die Koordinaten falsch aus; man muss sich in lat-lon abhängig vom Punkt an dem man startet bewegen.

    tiles = []
    center_lat, center_lon = center

    # Convert tile size from kilometers to degrees
    tile_size_deg = (tile_size_km / 111.12)  # Approximate conversion (1 degree = 111.12 km)

    # Calculate the number of tiles in each direction
    grid_size = int(grid_size_km / tile_size_km)

    # Calculate the offset to the top-left corner of the grid
    lat_offset = (grid_size - 1) / 2 * tile_size_deg
    lon_offset = -lat_offset  # Assuming longitude increases to the east

    # Generate the tiles
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the coordinates of each tile
            tile_lat = center_lat + lat_offset - i * tile_size_deg
            tile_lon = center_lon + lon_offset + j * tile_size_deg
            tiles.append((tile_lat, tile_lon))

    tile_lat = np.array([lat[0] for lat in tiles])
    tile_lon = np.array([lon[1] for lon in tiles])

    g = np.stack([tile_lat[::-1], tile_lon], 1)
    print(f'grid spanning from {g[0]} to {g[-1]} ...')

    return g



def _larger_matrix(a, b):
    """
    returns matrix r which shows if a-entries are larger than x-entries. example: r[2, 3] = True means a[2] > b[3]
    """
    a = np.reshape(a, (len(a), 1))
    b = np.reshape(b, (1, len(b)))
    return (a - b) > 0



def cell_coordinates(particle_latlonlocations: np.ndarray, grid: np.ndarray, _check_validity=True):
    """
    Given a numpy array of dimension (n, 2) which contains the x,y locations of the particles, this method returns in which grid cells each particle lies.
    Note: This method assumes the grid cells to be saved in the format of two numpy arrays, one containing the lat coordinates, the other the lon coordinates of the lower left corner of each tile.
    Args:
        particle_latlonlocations: np.ndarray of dimension (n, 2). particle_latlonlocations[k, :] = lat, lon of particle k.
        grid:
    Returns:
        # TODO
    """

    lat_locations = _larger_matrix(grid[:, 0], particle_latlonlocations[:, 0])
    lat_locations = np.argmax(lat_locations, 0)

    lon_locations = _larger_matrix(grid[:, 1], particle_latlonlocations[:, 1])
    lon_locations = np.argmax(lon_locations, 0)
    
    # Korrektur, da mit dieser Berechnung immer ein tile zu weit raus kommt:
    lat_locations -= np.sqrt(grid.shape[0]).astype(np.int32)
    lon_locations -= 1

    # one row of this is: (the index of the) largets lat location to which a particle belongs and (the index of the) largest lon location.
    #return np.stack([lat_locations, lon_locations], 1)
    
    if _check_validity:
        assert np.any((lat_locations + lon_locations) > 0)
    # this returns the index of the tile to which a particle belongs in the 'grid' array given to this method.
    return lat_locations + lon_locations


def is_in_same_cell(grid: np.ndarray, pos1: np.ndarray, pos2: np.ndarray):
    stacked_coordinates = np.stack([pos1, pos2], axis=0)
    stacked_indices = cell_coordinates(stacked_coordinates, grid)

    return np.all(stacked_indices[0] == stacked_indices[1])


def get_step_size(grid: np.ndarray):
    step_y = grid[1, 1] - grid[0, 1]
    idx_step_size = sqrt(grid.shape[0])
    assert idx_step_size == int(idx_step_size)
    step_x = grid[int(idx_step_size), 0] - grid[0, 0]

    return np.array((step_x, step_y))


def get_center_of_tile(point:np.ndarray, grid: np.ndarray):
    """
    Args:
        point: shape (2,), lat-lon coordinates of point for which we would like the index of the tile it belongs to.
    """
    expanded_pt = point[None, :]  # shape = (1, 2)
    tile_idx = cell_coordinates(expanded_pt, grid).squeeze()
    step_size = get_step_size(grid)

    return grid[tile_idx] + (step_size / 2)


def get_contours_of_tile(point:np.ndarray, grid:np.ndarray):
    """
    Returns left upper and right lower corner of point. This is different from the grid where each tile is defined by its left _lower_ corner.
    """
    tl = cell_coordinates(point[None, :], grid).squeeze()
    tl = grid[tl]
    #print('tl',tl )
    step_size = get_step_size(grid)

    left_upper = tl + np.array([0., step_size[1]])
    right_lower = tl + np.array([step_size[0], 0.])

    return left_upper.squeeze(), right_lower.squeeze()


#def is_in_grid(point:np.ndarray, grid:np.ndarray):
#    x, y = point
#
#    y_smaller_cond = y < grid[:, 0]
#    y_larger_cond = grid[:, 0] < y
#    x_smaller_cond = x < grid[:, 1]
#    x_larger_cond = grid[:, 1] < x
#
#    return np.any(y_smaller_cond) & np.any(y_larger_cond) & np.any(x_smaller_cond) & np.any(x_larger_cond)
def get_neighbouring_points(point:np.ndarray, grid:np.ndarray):
    # TODO: use numpy operations for this method
    step_size = get_step_size(grid)
    # TODO: in the future, eliminate get_center_of_tile call as we assume to only get centers.
    #cp = get_center_of_tile(point, grid)
    cp = point
    steps = [np.array((step_size[0], 0)),
             np.array((-step_size[0], 0)),
             np.array((0, step_size[1])),
             np.array((0, -step_size[1]))]
    #steps = [np.array((0, step_size[1])),
    #         np.array((-step_size[0], 0)),
    #         np.array((step_size[0], 0)),
    #         np.array((0, -step_size[1]))]
    #tiles = [cp + s for s in steps]
    tiles = cp + np.array(steps)

    positive_idxs = cell_coordinates(tiles, grid, _check_validity=False) >= 0
    #tiles = [t for t in tiles if is_in_grid(t, grid)]
    tiles = tiles[positive_idxs]
    assert len(tiles)


    ####
    #debug
    position = point
    #step_size = get_step_size(grid)[0]
    directions = (tiles - position) / step_size
    if len(directions) == 4 and (np.abs(np.sum(np.abs(directions)) - 4 ) > 1e-5):
        print('scheise (utils/grid):')
        print(directions)
    ####

    return tiles


if __name__ == '__main__':
    grid_center = (53.925052, 7.192578)
    uav_pos = np.array([53.72328, 7.20801])
    g = generate_grid(grid_center, 0.1, 50)
    print(g)
    print(g.shape)

    print(get_contours_of_tile(np.array(uav_pos), g))
    
    fifthneighbour = get_neighbouring_points(get_neighbouring_points(np.array(grid_center), g)[1], g)[2]
    print('normalized step: ', (np.array(grid_center)-fifthneighbour)/get_step_size(g))
    sys.exit(1)


    # check, whether get_contours_of_tile and get_center_of_tile are mutually 'inverse'.
    # does work as expected; note, that get_contours_of_tile returns tiles in a different format that they are saved in the grid.

    print('\n')
    idx = 1280
    print(f'tile from grid: {g[idx]}\ncontours of center: {get_contours_of_tile(get_center_of_tile(g[1280], g), g)}')
    print(f'calculating grid point from contours: ')
    cont = get_contours_of_tile(get_center_of_tile(g[1280], g), g)
    pt = np.array((cont[0][0], cont[1][1]))
    print(f'grid point from start is equal to calculated grid point: {np.all(g[idx] == pt)}')

    print('\ncalculating grid point from middle point of contours:')
    c = get_center_of_tile(get_contours_of_tile(g[idx], g)[1], g)
    gpt = g[cell_coordinates(c[None, :], g)]
    print(f'c:{c}    gpt:{gpt}')
    print(f'grid point from start is equal to calculated grid point: {g[idx] - gpt}')
    
    
    print(f'neighbouring points of {g[idx]}:', get_neighbouring_points(g[idx], g))
    print(f'difference of these to the point: {[get_center_of_tile(g[idx], g) - n for n in get_neighbouring_points(g[idx], g)]}')


    sys.exit(1)

    #lat_min, lat_max = np.min(g[:, 0]), np.max(g[:, 0])
    #lon_min, lon_max = np.min(g[:, 1]), np.max(g[:, 1])

    #latlon = np.random.rand(20, 2)
    #latlon[:, 0] = lat_min + (latlon[:, 0] * (lat_max - lat_min))
    #latlon[:, 1] = lon_min + (latlon[:, 1] * (lon_max - lon_min))
    ##print('\n', latlon)

    #print('\n', cell_coordinates(particle_latlonlocations=latlon, grid=g))

    pt = np.array([0.73227, 0.208])

    print(get_step_size(g))
    s = get_step_size(g)
    print(g[3, 0] - g[2, 0] - s[0], g[1, 1] - g[0, 1] - s[1])
    print()
    
    c = get_contours_of_tile(pt, g)
    c = np.abs(c[0] - c[1]) / 2
    print(c)
    coords = cell_coordinates(pt[None, :], g).squeeze()
    coords = g[coords]
    print('manually calced center', coords + c)
    print('function calculated center', get_center_of_tile(pt, g))
    print('difference:', coords + c - get_center_of_tile(pt, g))

