# pathplanningrepository

This repository provides code for the paper [Evaluating UAV Path Planning Algorithms for Realistic Maritime Search and Rescue Missions
](https://arxiv.org/abs/2402.01494). It can run simulations for UAV-based maritime search and rescue missions.
These simulations generate search targets at predefined user-specified locations in the sea, accompanied by particles sampled around them to emulate uncertainty about the precise target location.. For the drift simulation we employ [OpenDrift](https://github.com/OpenDrift/opendrift), which is an involved simulation software for ocean drift employing weather data like water current and wind flow.
In this readme, we will provide basic instructions on how to install the repository and give basic examples on how to run it.

## Install
1. Clone this repository:<br>
	```git clone https://github.com/cogsys-tuebingen/pathplanningrepository && cd pathplanningrepository```
2. Install OpenDrift:<br>
	```git clone https://github.com/OpenDrift/opendrift.git && cd opendrift```<br>
	Follow [their installation instructions](https://opendrift.github.io/install.html).


The dependencies required by OpenDrift include all necessary packages for this repository.


## Command Line Examples
To run a simulation, call ```python3 run_drone_simulation.py``` with one of the following arguments:
| Argument        | Description                             |
|-----------------|-----------------------------------------|
| `--config_file`  | File containing most configurations.               |
| `--log_file_name`| Name of file for saving all the logs and stats.       |
| `--no_plot`| If 0, a plot-like video file of the drone's trajectory will be created. If 1, it does not plot at all. Defaults to 0.          |
| `--verbose`     | Enables verbose logging                 |

For example, a valid command looks like this:<br>
```python3 run_drone_simulation.py --config ./example_configs/8-23_recbnb550_distance30_duration2_1532.json --log_file_name test_run --no_plot 1```

Example files for the configurations are given in the directory ```./example_configs```.
Please note, that all relevant locations (e.g. ```initial_position```, each of the search targets) should be wide within the grid to not run into bugs. Please note, that if you run into this error:
```
ValueError: Simulation stopped within first timestep. "Missing variables: ['x_sea_water_velocity', 'y_sea_water_velocity']", 'The simulation stopped before requested end time was reached.'
```
Then, most probably, you entered a date, time, or location for which weather data is unavailable.
Possible values for 'type' (under 'agent') are 'recbnb', 'spiral', and 'rectangle', resulting in the three types described in the paper. Under the current configurations, one hour in real time is 648 time steps in simulation (The UAV is simulated at a speed of 18 meters per second).
