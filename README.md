<img src="assets/platoon_icon_cropped_no_background.png" width="400">

_Note: Like much else in this repo, documentation is incomplete and WIP._

## Setup

Install core platoon package:
```bash
# In the root directory of the repo
uv sync
```

Install a plugin or an extension:
```bash
cd plugin-root-directory
uv sync
``` 

## Training a model with Reinforcement Learning
Make appropriate chagnes to the config file used in the command(s) below.
We use AReaL for our reinforcement learning backend. Please refer to its documentation for AReaL installation and config options: https://github.com/inclusionAI/AReaL/tree/main. 

Single Node Training Example:
```bash
cd plugins/textcraft # replace with your plugin of choice
uv run python3 -m areal.launcher.local /mnt/efs/platoon/plugins/textcraft/platoon/textcraft/train.py --config /mnt/efs/platoon/plugins/textcraft/platoon/textcraft/textcraft_reinforce_plus_plus.yaml experiment_name=textcraft-reinforce trial_name=trial0 > logs.md
```

## Visualizing Trajectories

See the dedicated guide: [Trajectory visualization CLI](src/platoon/visualization/README.md).
