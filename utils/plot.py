from pickle import TRUE
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import datetime
import math

from pyproj.transformer import TransformerUnsafe
from utils.random_crap import get_simulation_time, meter_distance_from_latlon

# Function to create a matplotlib plot (replace this with your custom plot)
def create_plot(frame_number):
    plt.clf()  # Clear the current figure
    # Your plot code here
    # Example:
    plt.plot(range(frame_number+1), 'r-')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Frame: {frame_number}')
    return plt

def _stretch(factor:int, array):
    r = list()
    for a in array:
        for _ in range(factor):
            r.append(a)

    return np.array(r)


def _strip_array(arr):
    """
    Strip the input array by removing consecutive time steps where all elements remain unchanged.
    Parameters:
    arr (numpy.ndarray): Input array of shape (T, N, 5)
    Returns:
    numpy.ndarray: Cleaned array with only time steps where changes occur
    """
    # Initialize a list to store the cleaned array
    cleaned_list = []
    
    # Iterate through the time steps
    for t in range(arr.shape[0]):
        # If it's the first time step, add it to the cleaned list
        if t == 0:
            cleaned_list.append(arr[t])
        else:
            # Compare the current time step with the previous one
            if not np.array_equal(arr[t, :, 1:3], arr[t-1, :, 1:3]):
                cleaned_list.append(arr[t])
    
    # Convert the cleaned list back to a numpy array
    cleaned_array = np.array(cleaned_list)[:, :, 1:3]
    
    return cleaned_array


"""
it's simulation time, not human time
"""
def _get_human_time(t):
    global fps_factor
    one_hour = fps_factor * 647
    h = int(t / one_hour)
    m = int(((t % one_hour) / one_hour) * 60)
    m = f'0{m}' if m < 10 else str(m)

    return f'{h}:{m}'


def _get_human_minutes(t):
    s = _get_human_time(t)
    h, m = s.split(':')
    return int((60 * int(h)) + int(m))


def _detect_position_at_first_moment_of_detection(p):
    """
    p: array of particles as we usually have it here, INCLUDING the time dimension.
    detects the first time a superparticle is detected and returns the position at which this happens.
    if that never happens, it returns the initial position of one of the superparticles.
    """
    is_superparticle = p[0, :, 0] == p[0, :, 4]
    sup = p[:, is_superparticle]
    
    min_discovered_time = int(1e9)
    min_idx = int(1e9)
    for _sps in range(len(sup[0])):
        is_discovered = sup[:, _sps, 3]
        discovered_time = is_discovered.nonzero()[0][0]
        if discovered_time < min_discovered_time:
            min_discovered_time = discovered_time
            min_idx = _sps
    
    if min_discovered_time > 1e8:
        return p[0, 0, 1:3]

    return sup[min_discovered_time, min_idx, 1:3]


def _detect_position_at_first_particle_encounter(p):
    """
    p: array of particles as we usually have it here, INCLUDING the time dimension.
    detects the first time a superparticle is detected and returns the position at which this happens.
    if that never happens, it returns the initial position of one of the superparticles.
    """
    is_discovered = p[:, :, 3]

    discovered_list = [np.any(is_discovered[k, :]) for k in range(is_discovered.shape[0])]
    discovered_time = discovered_list.index(True)
    discovered_idx = (is_discovered[discovered_time] == 1).nonzero()[0][0]

    return p[discovered_time, discovered_idx, 1:3], discovered_time
    



def _wrapper(coordinates, uav_position_nodes=None, fps=10, **kwargs):
    #lat_min, lat_max = np.min(coordinates[:, :, 1]), np.max(coordinates[:, :, 1])
    #lon_min, lon_max = np.min(coordinates[:, :, 2]), np.max(coordinates[:, :, 2])
    coordinates[np.isnan(coordinates)] = 300
    # TODO: extract min and max automatically and find out, why there are nans
    lat_min, lat_max = 53.6, 54.2
    lon_min, lon_max = 6.8, 7.7
    #lat_min, lat_max = 53.69, 53.8
    #lon_min, lon_max = 7.1, 7.25
    #lat_min, lat_max = 53.72, 53.74
    #lon_min, lon_max = 7.16, 7.19

    lat_difference, lon_difference = 53.74-53.72, 7.19-7.16

    plot_trajectory = kwargs.get('plot_trajectory')
    plot_trajectory = True if plot_trajectory else False
    
    if isinstance(uav_position_nodes[0], np.ndarray):
        np_uav_positions = uav_position_nodes
    else:
        np_uav_positions = [p.position for p in uav_position_nodes] if uav_position_nodes else []


    REFERENCE_FPS = 10

    stretch_factor = 1
    if fps != 10:
        stretch_factor = fps // REFERENCE_FPS

        np_uav_positions = _stretch(stretch_factor, np_uav_positions)
        coordinates = _stretch(stretch_factor, coordinates)
        uav_position_nodes = _stretch(stretch_factor, uav_position_nodes)



    max_time = len(np_uav_positions)

    _stripped_particles = _strip_array(coordinates)


    plot_settings = kwargs.get('plot_settings', dict())
    # Default is the position of the first time a superparticle is found.
    plot_position = plot_settings.get('plot_position', None)

    if plot_position == 'first_particle_encounter':
        plot_position, discovered_time= _detect_position_at_first_particle_encounter(coordinates)
        # use one super particle as example to determine distance; assert it's actually a superparticle
        assert coordinates[0, 0, 0] == coordinates[0, 0, 4]
        dist = meter_distance_from_latlon(np.array(kwargs['agent']['agent_settings']['initial_position']),
                                          np.array(coordinates[0, 0, 1:3]))
        print(f'distance while plottin: {dist} meters ...')
        if 9000 < dist < 11000:
            time_offset = 80
        elif 19000 < dist < 21000:
            time_offset = 90
        elif 4000 < dist < 6000:
            time_offset = 75

        plot_position = np_uav_positions[discovered_time+time_offset]
    elif not plot_position or plot_position == 'first_superparticle_found':
        plot_position = _detect_position_at_first_moment_of_detection(coordinates)

    """
    TEMP"""
    temp_x, temp_y = np.random.normal(7.210, scale=0.005, size=50), np.random.normal(53.8, scale=0.005, size=50)



    #time_translation = 1.
    #if np_uav_positions:
    #    time_translation = len(coordinates) / len(np_uav_positions)

    plot_size = 1
    # roughly 100 m:
    uav_plot_size = 1.1e-3

    
    global log_string
    log_string = ''

    simulation_interval_minutes = kwargs['simulation_interval_minutes']
    tile_size_km = kwargs['tile_size_km']


    # TODO: this method does not work at all
    def _plot_text(points, idx_discovered):
        # Filtern der Punkte basierend auf den entdeckten Indizes
        discovered_points = points[idx_discovered, 4]

        # Berechnung der Häufigkeit der entdeckten Punkte
        unique_discovered_points, counts = np.unique(discovered_points, return_counts=True)
        
        # Erstellen eines Dictionaries, um die Häufigkeiten zuzuordnen
        frequency_dict = dict(zip(unique_discovered_points, counts))

        #if len(frequency_dict):
        #    print('break in plot')

        for m, (k, v) in enumerate(frequency_dict.items()):
            s = f'{v} discovered for {k}\n'
            
            global log_string
            log_string += s

            plt.text((lon_max - lon_min)/2, (lat_max - lat_min)/2, s, fontsize=12)
        plt.text((lon_max - lon_min)/2, (lat_max - lat_min)/2, 'yesyesjoa', fontsize=12, ha='center')
        plt.text((lat_max - lat_min)/2, (lon_max - lon_min)/2, 'yesyesjoa(transp)', fontsize=12, ha='center')


    def plot_points_with_colors(frame_number):
        SMOOTH_ANIMATION = False
        plt.clf()
        
        frame_number = max(0, frame_number-1)
        #coord_idx = int(time_translation * frame_number)-1

        #coord_idx = get_simulation_time(frame_number, simulation_interval_minutes, tile_size_km)
        coord_idx = frame_number 

        # TODO: replace by suitable tqdm
        if frame_number % 100 == 1:
            print(f'plotting at {-1+frame_number}')
            #print(f'coord_idx = {coord_idx}')
        
        points = coordinates[coord_idx]
        #points = coordinates
        #uav_idx = min(len(np_uav_positions)-1, frame_number+10)
        uav_idx = frame_number
        global log_string
        log_string += f'uav_time = {uav_idx}, sim_time = {coord_idx}\n'

        uav_pos = np_uav_positions[uav_idx]
        log_string += f'uav_pos = {uav_pos}\n'
        #uav_pos = np_uav_positions[min(len(np_uav_positions)-1, frame_number+2)]

        # Extract x and y coordinates from the numpy array
        y_coords = points[:, 1]
        x_coords = points[:, 2]

        # Extract the first index (ID) and the fourth index
        ids = points[:, 0]
        discovered = points[:, 3]
        fourth_index = points[:, 4]

        #for i in range(len(points)):
        #    plt.scatter(x_coords[i], y_coords[i], color='black')
        #    if discovered[i] == 1:
        #        plt.scatter(x_coords[i], y_coords[i], color='gray')
        #    if ids[i] == fourth_index[i]:
        #        plt.scatter(x_coords[i], y_coords[i], color='red')
        
        # Plot points with different colors based on conditions
        idx_discovered = discovered == 1
        idx_superparticle = ids == fourth_index
        idx_superparticle &= ~ idx_discovered
        idx_alltherest = ~(idx_discovered | idx_superparticle)

        if not SMOOTH_ANIMATION:
        #if not SMOOTH_ANIMATION or True:
            _x, _y = x_coords, y_coords
        else:
            _pts = _interpolate_points(_stripped_particles, uav_idx, len(uav_position_nodes))

            _x, _y = _pts[:, 1], _pts[:, 0]


        ##plt.scatter(x_coords[idx_discovered], y_coords[idx_discovered], color='gray', s=plot_size)
        #plt.scatter(x_coords[idx_alltherest], y_coords[idx_alltherest], color='black', s=plot_size)
        ## TODO: discovered ones should be gray points and stop moving, also
        #plt.scatter(x_coords[idx_superparticle], y_coords[idx_superparticle], marker='o', color='red', s=7)
        #plt.scatter([500], [500], color='black', s=plot_size)

        PULSE = True
        PINK = (np.array([252, 15, 192])/255).tolist()
        _give_size = lambda x: 0
        if PULSE:
            _give_size = lambda x: int(8 - (x % 16))
        
        if True:
            #plt.scatter(x_coords[idx_discovered], y_coords[idx_discovered], color='gray', s=plot_size)
            plt.scatter(_x[idx_alltherest], _y[idx_alltherest], color='black', s=plot_size)
            # TODO: discovered ones should be gray points and stop moving, also
            #plt.scatter(_x[idx_superparticle], _y[idx_superparticle], marker='o', color='red', s=11 + _give_size(uav_idx))
            plt.scatter(_x[idx_superparticle], _y[idx_superparticle], marker='o', color=PINK, s=21 + _give_size(uav_idx))
            plt.scatter([500], [500], color='black', s=plot_size)
        else:
            plt.scatter(_x, _y, color='black', s=plot_size)



        #plt.scatter(uav_pos[1], uav_pos[0], color='green')

        # this crap doesn't work at all
        _plot_text(points, idx_discovered)

        
        ax = plt.gca()


        # plot discovered particles:
        l = []
        _l_colours = []

####################################
# all info plotting:
        l.append('{:.4f}, {:.4f}, t={}'.format(uav_position_nodes[uav_idx].position[0], uav_position_nodes[uav_idx].position[1], _get_human_time(uav_position_nodes[uav_idx].time)))
        _l_colours.append('black')
        for info in uav_position_nodes[uav_idx].particle_info[:, :2]:
            idsp, num_disc = info.astype(np.int32)
            l.append(f'{idsp}: {num_disc}')

            # find corresponding sp:
            sp = points[points[:, 0] == idsp]
            discovered = sp[:, 3] == 1
            _l_colours.append('gray' if discovered else 'black')

###################################


        #l.append('t={}'.format(_get_human_time(uav_position_nodes[uav_idx].time)))

        l = ax.legend(l)

        # set colour of text in legend:
        for text, colour in zip(l.get_texts(), _l_colours):
            text.set_color(colour)


        #line_x = np.array([52.9, 52,8009, 52,8009, 52,9])
        #line_y = np.array([7.9, 7.9009, 7,9009, 7,9])
        #plt.plot(line_x, line_y, color='blue')
        #rec = patches.Rectangle((7.1, 53.9), 0.009, 0.009, linewidth=1, edgecolor='blue', facecolor='none')
        # Rechteck mit 100 Meter Seitenlänge:
        # (two colours so its visible in front of white and black background; would be better if drawn adaptively)
        karminrot = np.array((165., 30., 55.))
        r = uav_plot_size
        rec_bl = patches.Rectangle((uav_pos[1]-r/2, uav_pos[0]-r/2), r, r, linewidth=1, edgecolor=(karminrot/255).tolist(), facecolor='none')
        rec_white = patches.Rectangle((uav_pos[1]-(r/2-r/8), uav_pos[0]-(r/2+r/8)), r, r, linewidth=1, edgecolor='white', facecolor='none')
        ax.add_patch(rec_bl)
        ax.add_patch(rec_white)

        if plot_trajectory:
            get_color = lambda x: (((1.-x)*np.array([255., 255., 255.]) + x*karminrot)/255).tolist()
            for k, p in enumerate(np_uav_positions[:max(0, uav_idx-1)]):
                rec_blass = patches.Rectangle((p[1]-(r/2-r/8), p[0]-(r/2+r/8)), r, r, linewidth=1, edgecolor='white', facecolor=get_color(k/uav_idx))
                ax.add_patch(rec_blass)



        # Add labels and show the plot
        plt.ylabel('latitude')
        plt.xlabel('longitude')
        #plt.title('Points with Different Colors')
        a = kwargs['agent']['type']
        n = {'spiral': 'Spiral Agent',
             'rectangle': 'Rectangle Agent',
             'recbnb': 'B&B Agent'}
        plt.title(f'SAR Mission {n[a]}')
        
        
        STATIC_VIEW = True
        _this_stretch = 50
        if lat_difference and lon_difference:
            #if SMOOTH_ANIMATION and ((uav_idx + stretch_factor) < max_time):
            if SMOOTH_ANIMATION and ((uav_idx + _this_stretch) < max_time):
                #next_pos = np_uav_positions[uav_idx+stretch_factor]
                next_pos = np_uav_positions[_this_stretch*(uav_idx//_this_stretch)+_this_stretch]
                former_pos = np_uav_positions[_this_stretch*(uav_idx//_this_stretch)]
                #inter_pos = _inter_pos(uav_pos, next_pos, uav_idx, stretch_factor)
                inter_pos = _inter_pos(former_pos, next_pos, uav_idx, _this_stretch)
                lon_min, lon_max = inter_pos[1]-lon_difference/2, inter_pos[1]+lon_difference
                lat_min, lat_max = inter_pos[0]-lat_difference/2, inter_pos[0]+lat_difference
            else:
                # hier kommen die Verschiebung des UAV im Plot her:
                lon_min, lon_max = uav_pos[1]-lon_difference/2, uav_pos[1]+lon_difference
                lat_min, lat_max = uav_pos[0]-lat_difference/2, uav_pos[0]+lat_difference


            if STATIC_VIEW:
                #lon_c, lat_c = 7.210, 53.8
                lat_c, lon_c = plot_position
                lon_min, lon_max = lon_c - lon_difference, lon_c + lon_difference
                lat_min, lat_max = lat_c - lat_difference, lat_c + lat_difference


        #print(f'lon min/max: {lon_min}     {lon_max}')
        plt.xlim(lon_min, lon_max)
        plt.ylim(lat_min, lat_max)

        log_string += '\n\n'
        return plt
    return plot_points_with_colors


def _inter_pos(current_pos, next_pos, uav_idx, stretch_factor):
    x = current_pos[0] + (uav_idx % stretch_factor) * (next_pos[0] - current_pos[0]) / stretch_factor
    y = current_pos[1] + (uav_idx % stretch_factor) * (next_pos[1] - current_pos[1]) / stretch_factor

    return x, y



def _create_time_lookup(local_time:int, max_time:int, rd=dict()) -> dict:
    # default arguments are created once, not every time the function is called;
    # so we can save the dict in the default argument.
    # TODO: that doesnt work at all. its calculated all over again, each time.
    if rd:
        return rd[local_time]
    # initialize dict to contain coarse idx values without interpolating t:
    rd = {k: _get_human_minutes(k) for k in range(max_time)}

    # calculate t now:
    def _get_next_step_index(idx):
        for k in range(1+idx, len(rd)):
            if rd[k] == rd[idx]:
                continue
            return k-idx

        return len(rd)-idx

    fine_step = 0
    step_idx = 0
    for k in range(max_time):
        if k == 0 or rd[k] != rd[k-1][0]:
            fine_step = 1 / _get_next_step_index(k)
            step_idx = 0
        rd[k] = (rd[k], step_idx * fine_step)
        step_idx += 1

    return rd[local_time]


def _interpolate_points(coords, time_idx_fine, max_time):
    """
    coords: numpy.ndarray of shape (T, N, 2), where T is the maximum time in the simulation frame
    """
    INTERPOLATE = True
    #idx = _get_human_minutes(time_idx_fine)
    idx, t = _create_time_lookup(time_idx_fine, max_time)

    if (not INTERPOLATE) or (idx >= len(coords)):
        return coords[idx]

    #global fps_factor
    #INTERVAL_LENGTH = 1+math.ceil(fps_factor * 647 / 60)
    #t = (time_idx_fine % INTERVAL_LENGTH) / INTERVAL_LENGTH

    # for debugging:
    #print(f'idx = {idx}, t = {t}, idx+t = {idx+t} (INTERVAL_LENGTH = {INTERVAL_LENGTH})\n')
    #print(f'idx = {idx}, t = {t}, idx+t = {idx+t}\n')

    return t*coords[idx + 1] + (1-t)*coords[idx]


def get_current_time():
    current_time = datetime.datetime.now().time()
    formatted_time = current_time.strftime("%H:%M:%S")
    return formatted_time


def create_animation(output_video_filename, coordinates, uav_position_nodes, fps=10, **kwargs):
    # Create a video from frames
    num_frames = len(uav_position_nodes) if uav_position_nodes else len(coordinates) # Number of frames to create
    global fps_factor
    fps = 30  # Frames per second for the video
    fps_factor = fps / 10
    if fps != 10:
        num_frames *= fps // 10
    
    """TEMP"""
    #num_frames = min(num_frames, 450)
    """TEMP"""
    # plot_trajectory needs is expected to be 1 or 0.
    plot_trajectory = kwargs.pop('plot_trajectory', True)
    plot_trajectory = bool(plot_trajectory)

    # Create the animation
    wrapper = _wrapper(coordinates, uav_position_nodes, fps, simulation_interval_minutes=kwargs['settings']['simulation_interval_minutes'], tile_size_km=kwargs['grid_settings']['tile_size_km'], plot_trajectory=plot_trajectory, **kwargs)
    animation = FuncAnimation(plt.figure(), wrapper, frames=num_frames, interval=1000 // fps)

    # Save the animation as a video
    animation.save(output_video_filename, fps=fps, dpi=100)

    with open(f'/tmp/log_{get_current_time()}.txt', 'w+') as f:
        f.write(log_string)


    print(f"Video saved as: {output_video_filename}")
    
if __name__ == '__main__':
    points_array = np.array([2*[
        [1, 2, 3, 1],
        [2, 3, 4, 0],
        [3, 1, 2, 3],
        [4, 5, 6, 4],
        [5, 2, 5, 1],
        [6, 1, 9, 1],
        [7, 3, 1, 0],
    ]])


    create_animation('/tmp/output_video.mp4', points_array, None)


