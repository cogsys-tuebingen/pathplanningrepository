import argparse
import json
from datetime import datetime, timedelta, date
from os.path import exists
import time
import os

import numpy as np

from opendrift.models.leeway import Leeway

from utils.history import ParticleManager
from utils.node import Node
from utils.random_crap import read_config, get_adjusted_uav_time_steps, parse_time, get_simulation_time
from utils.grid import generate_grid, get_contours_of_tile, get_center_of_tile
from utils.plot import create_animation

from agents.factory import agent_factory
from utils.stats import make_stats_from_observation, write_stats_to_file


def parse_args():
    # TODO: Argument Parser
    parser = argparse.ArgumentParser("Call \'python3 run_drone_simulation.py\' with one of below arguments.")

    # maybe make required later.
    parser.add_argument('--config_file', type=str, default=None,
                        help='File containing most configurations.')
    parser.add_argument('--log_file_name', type=str, default='output',
                        help='Name of file for saving all the logs and stats.')
    parser.add_argument('--no_plot', type=int, default=0,
                        help='if 0, it plots. if 1, it does not plot.')

    args = parser.parse_args()
    if args.config_file is not None:
        file_config = read_config(args.config_file)
        args.file_config = file_config
        #args = convert_namespaces_to_dicts(args, file_config)

    return args


def run_simulation(opt):
    # TODO: run LeeWay simulation

    # loglevel = 20 means no debugging info, loglevel = 0 is the other extreme.
    o = Leeway(loglevel=20)
    o.add_readers_from_list([#'https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be',
                             'https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest',
                             'https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/ncep_global/NCEP_Global_Atmospheric_Model_best.ncd'])

    #o.add_readers_from_list([# replaces hawaii:
    #                         './resources/wind/ncep_global_NCEP_Global_Atmospheric_Model_best.nc',])
    #                         # the rest replaces hycom:
    #    './resources/water_current/hycom_glby_930_2023052712_t000_ssh.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t003_ssh.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t006_ssh.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t009_ssh.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t012_ssh.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t015_ssh.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t018_ssh.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t021_ssh.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t000_ts3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t000_uv3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t003_ts3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t003_uv3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t006_ts3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t006_uv3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t009_ts3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t009_uv3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t012_ts3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t012_uv3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t015_ts3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t015_uv3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t018_ts3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t018_uv3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t021_ts3z.nc',
    #                    './resources/water_current/hycom_glby_930_2023052712_t021_uv3z.nc',
    #                    './resources/water_current/hycom_GLBy0.08_930_2023052712_t000_ice.nc',
    #                    './resources/water_current/hycom_GLBy0.08_930_2023052712_t003_ice.nc',
    #                    './resources/water_current/hycom_GLBy0.08_930_2023052712_t006_ice.nc',
    #                    './resources/water_current/hycom_GLBy0.08_930_2023052712_t009_ice.nc',
    #                    './resources/water_current/hycom_GLBy0.08_930_2023052712_t012_ice.nc',
    #                    './resources/water_current/hycom_GLBy0.08_930_2023052712_t015_ice.nc',
    #                    './resources/water_current/hycom_GLBy0.08_930_2023052712_t018_ice.nc',
    #                    './resources/water_current/hycom_GLBy0.08_930_2023052712_t021_ice.nc'])
    
    # 26 is a Life-raft without ballast.
    # TODO experiment with other object types.
    object_type = 26

    num_particles = opt.file_config['coordinates'][0].get('num_particles')
    if num_particles is None:
        num_particles = 999

    PM = ParticleManager((1+num_particles) * len(opt.file_config['coordinates']), file_config=opt.file_config)

    ids_super_particle, range_following_particle_ids = list(), list()

    for c in opt.file_config['coordinates']:
        lon, lat, radius = c['lon'], c['lat'], c.get('seed_radius')

        if radius is None:
            radius = 1000
        # what unit does the radius carry?


        # first, seed the super particle:
        o.seed_elements(lon=lon, lat=lat,
                         radius=1,
                         number=1,
                         object_type=object_type,
                         time=parse_time(opt.file_config['time']))

        ids_super_particle.append(o.num_elements_scheduled())

        # next, seed the following particles:
        o.seed_elements(lon=lon, lat=lat, radius=radius,
                        number=num_particles,
                        object_type=object_type,
                        time=parse_time(opt.file_config['time']))
        range_following_particle_ids.append((1+ids_super_particle[-1], o.num_elements_scheduled()))


    d = opt.file_config['settings']
    duration, time_step = d['simulation_duration_hours'], d['simulation_interval_minutes']

    # TODO: synchronize args.uav_time_steps and number of time steps in simulation
    o.run(duration=timedelta(hours=duration),
          time_step=timedelta(minutes=time_step),
          time_step_output=timedelta(minutes=time_step))

    import time
    s = time.time()

    PM.initialize(o.history)
    print(f'\n\ninitializing took {time.time()-s} seconds ...')
    
    s = time.time()
    PM.add_particles(ids_super_particle, range_following_particle_ids)
    print(f'\n\nparticle adding took {time.time()-s} seconds ...')

    return o, PM


def move(uav, s):
    # TODO
    # this method should move the UAV according to some motion law (no too sudden changes in velocity vector)
    # it can, if the uav_pos is expressed in terms of lat and lon, use a transformation of coordinates like proj4. (would add more realism, aka sell well)
    return uav

def get_super_particles(h,cutoff,t):
    # index of particles whose associated super particles are themselves (aka the super particles)
    b = h.particles[:, :, 0] == h.particles[:, :, 4]
    b &= h.particles[:, :, 1] < cutoff

    rp = h.particles[t][b[0].nonzero()]

    return rp

if __name__ == '__main__':
    args = parse_args()

    o, history = run_simulation(args)

    #grid_center = {'lat': 53.925052, 'lon': 7.192578} #TODO
    grid_center = (53.925052, 7.192578)
    #uav_pos = {'lat': 53.72328, 'lon': 7.20801}  # weisse duene
    #uav_pos = {'': 53.705, '2': 7.173387}
    #uav_pos = np.array(list(uav_pos.values()))
    grid_center = args.file_config['grid_settings']['grid_center']

    tile_size_km, grid_size_km = args.file_config['grid_settings']['tile_size_km'], args.file_config['grid_settings']['grid_size_km']  # standard: 0.1 and 25

    grid = generate_grid(grid_center, tile_size_km=tile_size_km, grid_size_km=grid_size_km)
    
    # this one is a grid point (center of a tile)
    #uav_pos = np.array([53.70456964, 7.17367951])
    #uav_pos = get_center_of_tile(uav_pos, grid)
    uav_time_steps = args.file_config['agent']['agent_settings']['uav_time_steps']


    adj_uav_time_steps = get_adjusted_uav_time_steps(uav_time_steps, file_config=args.file_config)


    #adj_uav_time_steps = 200


    if adj_uav_time_steps < uav_time_steps:
        args.file_config['agent']['agent_settings']['uav_time_steps'] = adj_uav_time_steps
        uav_time_steps = adj_uav_time_steps
        print(f'Adjusted uav_time_steps to {adj_uav_time_steps} to match the simulation time ...')

    s = time.time()
    history.adjust_particles(uav_time_steps)
    #print(f'adjusting took {time.time() - s} seconds ...')


    #agent = SpiralAgent(initial_position=uav_pos, **args.__dict__)
    #agent = BranchAndBoundAgent(initial_position=uav_pos,
    #                            grid=grid,
    #                            history=history,
    #                            uav_time_steps=args.uav_time_steps)
    agent_type = args.file_config['agent']['type']
    

    """
    TEMP
    """
    #if agent_type in ['recbnb', 'rec_bnb']:
    #    sys.exit(1)


    agent_settings = args.file_config['agent']['agent_settings']
    agent = agent_factory[agent_type](grid=grid, history=history, **agent_settings)

    agent.initialize(history, grid)
    
    stats = dict()
    stats['finding_time'] = dict()
    stats['percentage_finding_time'] = dict()

    uav_pos = agent.initial_position
    for k in range(uav_time_steps-1):
    #for k in range(200):
        #t = get_simulation_time(k, args.file_config['settings']['simulation_interval_minutes'], tile_size_km)
        t = k
        
        observation = history._get_observation(tile=get_contours_of_tile(uav_pos, grid), t=t, grid=grid)

        uav_pos_node = agent.step(observation, k)
        uav_pos = uav_pos_node.position if not isinstance(uav_pos_node, np.ndarray) else uav_pos_node
        if isinstance(uav_pos, Node) and isinstance(uav_pos.parent.parent, Node):
            if np.all(uav_pos.position == uav_pos.parent.parent.position):
                print(f'Node equal to Grandparent:\n{uav_pos}')

        if args.log_file_name:
            # TODO: why doesn't history.particles work?
            stats[k], found_sps = make_stats_from_observation(observation, uav_pos_node, history.particles[k], state=stats)
            if found_sps:
                for idx in found_sps:
                    _k = str(idx)
                    if not stats['finding_time'].get(_k):
                        stats['finding_time'][_k] = k
                        stats['percentage_finding_time'][_k] = k / adj_uav_time_steps



    if args.log_file_name:
        #print(stats)
        y, m, d = date.today().year, date.today().month, date.today().day
        METANAME = '0810-0817_5km'
        dir_name = f'./structured_experiments/{METANAME}/{y}-{m}-{d}/'
        os.makedirs(dir_name, exist_ok=True)
        fname = os.path.join(dir_name, args.log_file_name)
        write_stats_to_file(stats, fname, args.file_config)


            

    #create_animation('/tmp/output_video.mp4', history.particles, agent.position_history[(7*len(agent.position_history)//8):])
    s = time.time()

    fc = args.file_config.copy()
    del fc['coordinates']
    
    if args.no_plot == 0:
        y, m, d = date.today().year, date.today().month, date.today().day
        dir_name = f'./structured_experiments/{y}-{m}-{d}/videos/'
        os.makedirs(dir_name, exist_ok=True)
        #create_animation(f'/tmp/{args.log_file_name}_video.mp4', history.particles, agent.position_history, **fc)
        create_animation(os.path.join(f'{dir_name}', f'{args.log_file_name}_video.mp4'), history.particles, agent.position_history, **fc)
        print(f'creating animation took\n{time.time()-s} seconds ...')

    #o.animation(filename='/tmp/test_animation.mp4', fast=True, background=['x_sea_water_velocity', 'y_sea_water_velocity'])

    # TODO: plotting and logging

