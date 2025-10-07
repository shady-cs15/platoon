<img src="assets/platoon_icon_cropped_no_background.png" width="400">

_Note: Like much else in this repo, documentation is incomplete and WIP._

## Setup

With uv:
```
UV_GIT_LFS=1 uv sync --all-groups
```

With pip:
```
pip install -e .
```

## Training a model with Reinforcement Learning loop
Make appropriate chagnes to the config file used in the command below.
We use AReaL for our reinforcement learning backend. Please refer to its documentation for AReaL installation and config options: https://github.com/inclusionAI/AReaL/tree/main. 

Example:
```bash
python3 -m areal.launcher.local src/platoon/train/appworld_reinforce++.py --config src/platoon/train/appworld_reinforce++.yaml experiment_name=appworld-reinforce++ trial_name=trialreinforce0 > logs.md
```

## Visualizing Trajectories

See the dedicated guide: [Trajectory visualization CLI](src/platoon/visualization/README.md).
