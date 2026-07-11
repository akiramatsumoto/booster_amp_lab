# Booster RL Tasks
#

This repository is forked from [booster_train](https://github.com/BoosterRobotics/booster_rl_tasks.git), which is introduced AMP to implement natural running and walking in booster K1.
## Overview

This repository provides a set of reinforcement learning tasks for Booster robots using [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html).
Currently it includes the fabulous [BeyondMimic motion tracking](https://github.com/HybridRobotics/whole_body_tracking) framework adapted to Booster K1 robots.
This repository follows the standard Isaac Lab project structure, and is tested with IsaacLab 2.2 and Isaac Sim 5.0.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
    ```bash
    git clone https://github.com/BoosterRobotics/booster_rl_tasks.git
    ```

- Using a python interpreter that has Isaac Lab installed, install all dependencies
  (bundled `rsl_rl`, `booster_assets`, `ruamel.yaml`, and `booster_rl_tasks`) in one command
  from the repository root:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -r requirements.txt
    ```

  This installs the bundled `rsl_rl` and `booster_assets` in editable mode. If you instead
  keep `booster_assets` in a separate location, clone
  [booster_assets](https://github.com/BoosterRobotics/booster_assets) (Booster robot models and
  motion data) and install it from there following its own instructions.

- Prepare BeyondMimic motion data:
    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/csv_to_npz.py --headless --input_file=<PATH_TO_BOOSTER_ASSETS>/motions/K1/<MOTION>.csv --input_fps=<FPS> --output_name=<PATH_TO_BOOSTER_ASSETS>/motions/K1/<MOTION>.npz
    ```

## Usage

- Listing the available tasks:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/list_envs.py
    ```

- Running a task:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/rsl_rl/train.py --task=<TASK_NAME> --headless --device cuda:N
    ```

- Play a trained policy and export it for deployment:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/rsl_rl/play.py --task=<TASK_NAME> --checkpoint=<CHECKPOINT_PATH>
    ```

    This script also exports the trained policy to a TorchScript/ONNX file for deployment on real robots in `logs/rsl_rl/<EXPERIMENT>/<RUN>/exported/`.

## Deploy

After a model has been trained and exported, you can deploy the trained policy in MuJoCo or on real Booster robots using the [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repository. For more details, please refer to the instructions in the [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repository.


## Acknowledgements

- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): the motion tracking training in BeyondMimic, which is a versatile humanoid control framework that provides highly dynamic motion tracking.