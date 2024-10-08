import json
from logging import root
import os
import sys
from datetime import date, datetime



def _get_statistics(file_name:str, targets_available:int) -> dict:
    stat_dict = dict()

    with open(file_name, 'r') as f:
        full_dict = json.load(f)

    stat_dict['finding_time'] = full_dict['finding_time']
    stat_dict['targets_found'] = len(stat_dict['finding_time'])
    stat_dict['targets_available'] = targets_available

    #new_dict = {k: v for k,v in full_dict.items() if k not in ['finding_time', 'percentage_finding_time']}
    new_dict = full_dict.copy()
    del new_dict['finding_time']
    del new_dict['percentage_finding_time']

    l = [int(k) for k in new_dict]
    last_key = max(l)-1

    #total_particles, total_available = full_dict[str(last_key)]['overall'], full_dict[str(last_key)]['total_available']
    discovery_percentage = full_dict[str(last_key)]['particles']['overall']['percentage_discovered']

    stat_dict['particle_discovery_percentage'] = discovery_percentage

    return stat_dict



def make_global_statistics(stats_dict:dict):
    global_stats = dict()

    for _s in stats_dict.values():
        o_d, s = _s
        d = s['distance']
        if not global_stats.get(d):
            global_stats[d] = dict()
            for _k in ['rectangle', 'spiral', 'recbnb150', 'recbnb350', 'recbnb550']:
                global_stats[d][_k] = {'targets_discovered': 0,
                                       'targets_available': 0,
                                       'first_finding_time': 0,
                                       'second_finding_time': 0,
                                       'difference_finding_time': 0,
                                       'particle_discovery_percentage': 0,
                                       'total_runs': 0,
                                       'num_first_found': 0,
                                       'num_second_found': 0}

        agent = o_d['agent_type']
        if agent == 'recbnb':
            agent += str(o_d['local_steps'])
        
        global_stats[d][agent]['total_runs'] += 1
        global_stats[d][agent]['targets_discovered'] += s['targets_found']
        global_stats[d][agent]['targets_available'] += s['targets_available']
        if list(s['finding_time'].values()):
            first = min(s['finding_time'].values())
            global_stats[d][agent]['first_finding_time'] += first
            global_stats[d][agent]['num_first_found'] += 1
            if len(list(s['finding_time'].values())) > 1:
                second = max(s['finding_time'].values())
                global_stats[d][agent]['second_finding_time'] += second
                global_stats[d][agent]['difference_finding_time'] += second - first
                global_stats[d][agent]['num_second_found'] += 1

        
        global_stats[d][agent]['particle_discovery_percentage'] += s['particle_discovery_percentage']


    for _gs in global_stats.values():
        for agent_type in _gs:
            if not _gs[agent_type]['targets_available']:
                continue
            _gs[agent_type]['targets_discovered'] /= _gs[agent_type]['targets_available']
            if _gs[agent_type]['first_finding_time']:
                _gs[agent_type]['first_finding_time'] /= _gs[agent_type]['num_first_found']
            if _gs[agent_type]['second_finding_time']:
                _gs[agent_type]['second_finding_time'] /= _gs[agent_type]['num_second_found']
                _gs[agent_type]['difference_finding_time'] /= _gs[agent_type]['num_second_found']
            _gs[agent_type]['particle_discovery_percentage'] /= _gs[agent_type]['total_runs']


    return global_stats



def _extract_distance(fname:str):
    l = [_l for _l in fname.split('_') if 'distance' in _l]
    l = l[0]

    return int(l.replace('distance', ''))


def parse_json_files(root_dir:str, file_list:list) -> dict:
    overviews = {}
    
    for file_name in file_list:
        #file_name = os.path.join(root_dir, file_name)
        with open(os.path.join(root_dir, file_name), 'r') as f:
            data = json.load(f)
            overview_dict = dict()
            
            # Getting the length of the 'coordinates' list
            coordinates_length = len(data.get('coordinates', []))
            
            # Getting the 'time' value
            time_value = data.get('time', 'N/A')
            
            # Getting the agent's 'type'
            agent_type = data.get('agent', {}).get('type', 'N/A')
            
            # Getting the simulation's 'duration'
            duration = data.get('settings', {}).get('simulation_duration_hours', 'N/A')
            
            # Getting the targets' 'distance'
            distance = _extract_distance(file_name)

            # Getting the simulation's 'interval' for statistics calculation later
            simulation_interval_minutes = data.get('settings', {}).get('simulation_interval_minutes', 'N/A')
            
            # If agent's type is 'recbnb', getting the 'local_uav_time_steps'
            local_uav_time_steps = ''
            if agent_type == 'recbnb':
                local_uav_time_steps = data.get('agent', {}).get('agent_settings', {}).get('local_uav_time_steps', 'N/A')
            
            # Formatting the overview string
           # overview_str = f"Coordinates Length: {coordinates_length}, Time: {time_value}, Agent Type: {agent_type}"
            overview_str = f"num_targets: {coordinates_length}, time: {time_value}, agent_type: {agent_type}, duration: {duration}, distance: {distance}"
            overview_dict = {'num_targets': coordinates_length,
                             'time': time_value,
                             'agent_type': agent_type,
                             'duration': duration,
                             'distance': distance}
            if local_uav_time_steps:
                overview_str += f", local_steps: {local_uav_time_steps}"
                overview_dict['local_steps'] = local_uav_time_steps
            
            raw_data_file_name = os.path.join(root_dir, file_name).replace('_CONFIG', '') + '.json'
            stats_data = _get_statistics(raw_data_file_name, coordinates_length)
            stats_data['simulation_interval_minutes'] = simulation_interval_minutes
            stats_data['distance'] = distance

            # Adding the overview string to the overviews dict
            overviews[file_name] = {'o_s': overview_str,
                                    'o_d': overview_dict,
                                    'stats': stats_data}
            #overviews[file_name] = overview_str
    
    return overviews




def _parse_date(date_str):
    y, month, d, hm = date_str.split('-')
    h, minute = hm.split(':')
    return datetime(int(y), int(month), int(d), int(h), int(minute))


# Example usage
# file_list = ["file1.json", "file2.json", ...]
# print(parse_json_files(file_list))

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        root_dir = sys.argv[1]

    print(f'looking for result files in:\n{root_dir}\n')


    start_date = datetime(2024, 8, 1, 0, 0)
    end_date = datetime(2024, 7, 8, 0, 0)

    file_list = os.listdir(root_dir)
    file_list = [f for f in file_list if '_CONFIG' in f]

    parsed = parse_json_files(root_dir, file_list)


    # filter for desired dates:
    if end_date > start_date:
        print(f'Only evaluating experiments in between {start_date} and {end_date} ...')
        for k, v in list(parsed.items()):
            if not start_date <= _parse_date(v['o_d']['time']) <= end_date:
                del parsed[k]

    #for k, v in parsed.items():
    #    print(f'{k}: {v["o_s"]}')
    #    print(f'stats:\n{v["stats"]}\n\n')
    #print(parse_json_files(root_dir, file_list))
    
    for k, v in make_global_statistics({k: (v['o_d'], v['stats']) for k, v in parsed.items()}).items():
        print(f'distance: {k}')
        for agent_type, agent_stats in v.items():
            print(f'stats for {agent_type}: {agent_stats}\n\n')


