# LFMC-Gym



The training code is based on [Raisim](https://raisim.com/) and an adaptation of
[RaisimGymTorch](https://raisim.com/sections/RaisimGymTorch.html). The
environment is written in C++ while the actual training happens in Python. The current
example provided in the repository has been tuned to train a 25 Hz locomotion
policy for the [ANYmal C](https://youtu.be/_ffgWvdZyvk) robot in roughly 30 minutes
on a standard computer with an 8-core Intel i9-9900k CPU @ 3.6 GHz and an NVIDIA RTX 2080 Ti.

### Prerequisites
The training code depnds on [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
which can be installed in Ubuntu like so:
```console
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

```console
git clone --recurse-submodules git@github.com:ori-drs/lfmc_gym.git
```

Alternatively, you can clone the ```networks_minimal``` repository in 
the dependencies directory.
```console
cd lfmc_gym
git clone git@github.com:gsiddhant/networks_minimal.git dependencies/networks_minimal
```

We use the ```header``` branch of this repository.
```console
cd dependencies/networks_minimal
git checkout header
cd ../..
```

### Build

A ```setup.py``` script is handles the necessary dependencies
and C++ builds. It is recommended that you use a 
[virtual environment](https://docs.python.org/3/tutorial/venv.html).
If ```python3-venv``` is not already installed, use the following command.
```console
sudo apt-get install python3-venv
```

Assuming you are in the project root directory ```$LFMC_GYM_PATH```, 
create and source a virtual environment. 
```console
cd $LFMC_GYM_PATH
python3 -m venv venv
source venv/bin/activate
```

The relevant build and installs can then be done using
```console
python setup.py develop
```

### Usage
The environment provided is called ```anymal_velocity_command```
and is used to train a velocity command tracking locomotion policy.
Before you start training, launch the ```RaisimUnity``` visualizer and
check the Auto-connect option. The training can then be started using
```console
python scripts/anymal_velocity_command/runner.py
```

After about 5k training iterations, you should observed a 
good velocity tracking behavior. The checkpoints are stored
in the ```$LFMC_GYM_PATH/data/anymal_velocity_command/<date-time>```
directory. To test the trained policy, use the provided 
script.
```console
python scripts/anymal_velocity_command/tester.py
```
