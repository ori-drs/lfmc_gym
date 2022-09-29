# LFMC-Gym

Repository for training low-frequency motion control (LFMC) policies for 
robotic locomotion.

---

The training code is based on [Raisim](https://raisim.com/) and an adaptation of
[RaisimGymTorch](https://raisim.com/sections/RaisimGymTorch.html). The
environment is written in C++ while the actual training happens in Python. The current
example provided in the repository has been tuned to train a 25 Hz locomotion
policy for the [ANYmal C](https://youtu.be/_ffgWvdZyvk) robot in roughly 30 minutes
on a standard computer with an 8-core Intel i9-9900k CPU @ 3.6 GHz and an NVIDIA RTX 2080 Ti.

### Prerequisites
The training code depnds on [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
which can be installed in Ubuntu like so:
```bash
sudo apt-get install libeigen3-dev
```

You will also need to have installed 
Raisim. Please refer to the [Raisim documentation](https://raisim.com/sections/Installation.html) 
for install instructions. Based on the documentation, 
we will assume that Raisim has been installed in a directory
called ```$LOCAL_INSTALL```.

### Clone
To clone ```lfmc_gym```, use the following command. Note that, 
this repository depends upon a neural network implementation
written in C++ called [```networks_minimal```](https://github.com/gsiddhant/networks_minimal) 
and is included as a submodule. Ensure you
use ```--recurse-submodule``` flag while cloning the repository.

```bash
git clone --recurse-submodules git@github.com:ori-drs/lfmc_gym.git
```

Alternatively, you can clone the ```networks_minimal``` repository in 
the dependencies directory.
```bash
cd lfmc_gym
git clone git@github.com:gsiddhant/networks_minimal.git dependencies/networks_minimal
```

We use the ```header``` branch of this repository.
```bash
cd dependencies/networks_minimal
git checkout header
cd ../..
```

### Build

A ```setup.py``` script is handles the necessary dependencies
and C++ builds. It is recommended that you use a 
[virtual environment](https://docs.python.org/3/tutorial/venv.html).
If ```python3-venv``` is not already installed, use the following command.
```bash
sudo apt-get install python3-venv
```

Assuming you are in the project root directory ```$LFMC_GYM_PATH```, 
create and source a virtual environment. 
```bash
cd $LFMC_GYM_PATH
python3 -m venv venv
source venv/bin/activate
```

The relevant build and installs can then be done using
```bash
python setup.py develop
```

### Usage
The environment provided is called ```anymal_velocity_command```
and is used to train a velocity command tracking locomotion policy.
Before you start training, launch the ```RaisimUnity``` visualizer and
check the Auto-connect option. The training can then be started using
```bash
python scripts/anymal_velocity_command/runner.py
```

After about 5k training iterations, you should observed a 
good velocity tracking behavior. The checkpoints are stored
in the ```$LFMC_GYM_PATH/data/anymal_velocity_command/<date-time>```
directory. To test the trained policy, use the provided 
script.
```bash
python scripts/anymal_velocity_command/tester.py
```

### Code Structure
    └── common                          # Utilities used by modules throughout the project
        ├── paths.py                    # Utility to handle project related paths
    └── dependencies                    # Packages required by LFMC-Gym
        ├── actuation_dynamics          # Actuator network for ANYmal C
        ├── networks_minimal            # C++ based implementation of MLP and GRU networks
    └── gym_envs                        # RL environments
        ├── anymal_velocity_command     # The ANYmal-Velocity-Command gym environment
            └── Environment.hpp         # Describes the main RL Gym functions
            └── cfg,yaml                # Environment configuration and training parameters 
    └── modules                         # Actor-Critic network architectures and helper functions
        ├── ...
    └── raisim_gym_torch                # Python wrapper for Raisim-C++ env and PPO implementation
        ├── ...
    └── resources                       # Assets used in the project
        ├── models                      # Robot URDFs
        ├── parameters                  # Neural network parameters (eg. ANYmal C actuator network)
    └── scripts                         # Python scripts for training, evaluation and utilities
        ├── anymal_velocity_command     # Training and evaluation scripts for ANYmal-Velocity-Command
            └── runner.py               # Training script for ANYmal-Velocity-Command
            └── tester.py               # Evaluation script that executes trained policies 
        ├── utility                     # Scripts to extract model parameters
            └── ...
    └── CMakeLists.txt                  # C++ build utility
    └── setup.py                        # Executes C++ build and installs LFMC-Gym and dependencies

### Authors(s)
[Siddhant Gangapurwala](mailto:siddhant@robots.ox.ac.uk)
