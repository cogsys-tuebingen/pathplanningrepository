# pathplanningrepository

This repository provides code for the paper [Evaluating UAV Path Planning Algorithms for Realistic Maritime Search and Rescue Missions
](https://arxiv.org/abs/2402.01494). With it, you can run simulations for UAV-based maritime search and rescue missions.
These simulations generate search targets at predefined (by the user) locations in the sea together with some particles around it, emulating uncertainty about the precise target location. For the drift simulation we employ [OpenDrift](https://github.com/OpenDrift/opendrift), which is an involved simulation software for ocean drift employing weather data like water current and wind flow.
In this readme, we will provide basic instructions on how to install the repository and give basic examples on how to run it.

## Install
1. Clone the repostitory:<br>
	```git clone https://github.com/cogsys-tuebingen/pathplanningrepository && cd pathplanningrepository```
2. Install OpenDrift:<br>
	```git clone https://github.com/OpenDrift/opendrift.git && cd opendrift```<br>
	Follow [their installation instructions](https://opendrift.github.io/install.html).


## Command Line Examples
To run a simulation, one can call ```python3 run_drone_simulation.py``` with one of the following arguments:
| Argument        | Description                             |
|-----------------|-----------------------------------------|
| `--config_file`  | File containing most configurations.               |
| `log_file_name`| Name of file for saving all the logs and stats.       |
| `--no_plot`| If 0, a plot-like video file of the drone's trajectory will be created. If 1, it does not plot at all. Defaults to 0.          |
| `--verbose`     | Enables verbose logging                 |



