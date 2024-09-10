# TacGNN: Learning Tactile-Based In-Hand Manipulation With a Blind Robot 

[Website](https://sites.google.com/view/tacgnn) | [Technical Paper](https://bionicdl.ancorasir.com/wp-content/uploads/2023/11/2023-J-RAL-TacGNN.pdf) | [Videos](https://www.youtube.com/watch?v=nvTld4KoEiU)


### About this repository

This repository contains the experiments shown in [RAL paper](https://bionicdl.ancorasir.com/wp-content/uploads/2023/11/2023-J-RAL-TacGNN.pdf)


### Installation

Download the Isaac Gym Preview 3 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. We highly recommend using a conda environment 
to simplify set up.

Refer to the [Isaac Gym documentation](https://github.com/isaac-sim/IsaacGymEnvs) for more information.

### Running the benchmarks

To train the policy, run this:

```bash
python train.py task=AllegroHandBaodingGraph
num_envs=256
experiment=baoding_gnn
max_iterations=10000
headless=True
train.params.config.minibatch_size=2048
task.env.observationTouch=False
```

Use another perception model:

```bash
python train.py task=AllegroHandBaodingGraph
num_envs=256
experiment=baoding_mlp
max_iterations=10000
headless=True
train.params.config.minibatch_size=2048
task.env.touchmodel='mlp_model'
```

### Configuration and command line arguments

We use [Hydra](https://hydra.cc/docs/intro/) to manage the config. Note that this has some 
differences from previous incarnations in older versions of Isaac Gym.
 
Key arguments to the `train.py` script are:

* `task=TASK` - selects which task to use. Any of `AllegroHand`, `Ant`, `Anymal`, `AnymalTerrain`, `BallBalance`, `Cartpole`, `FrankaCabinet`, `Humanoid`, `Ingenuity` `Quadcopter`, `ShadowHand`, `ShadowHandOpenAI_FF`, `ShadowHandOpenAI_LSTM`, and `Trifinger` (these correspond to the config for each environment in the folder `isaacgymenvs/config/task`)
* `train=TRAIN` - selects which training config to use. Will automatically default to the correct config for the environment (ie. `<TASK>PPO`).
* `num_envs=NUM_ENVS` - selects the number of environments to use (overriding the default number of environments set in the task config).
* `seed=SEED` - sets a seed value for randomizations, and overrides the default seed set up in the task config
* `sim_device=SIM_DEVICE_TYPE` - Device used for physics simulation. Set to `cuda:0` (default) to use GPU and to `cpu` for CPU. Follows PyTorch-like device syntax.
* `rl_device=RL_DEVICE` - Which device / ID to use for the RL algorithm. Defaults to `cuda:0`, and also follows PyTorch-like device syntax.
* `graphics_device_id=GRAHPICS_DEVICE_ID` - Which Vulkan graphics device ID to use for rendering. Defaults to 0. **Note** - this may be different from CUDA device ID, and does **not** follow PyTorch-like device syntax.
* `pipeline=PIPELINE` - Which API pipeline to use. Defaults to `gpu`, can also set to `cpu`. When using the `gpu` pipeline, all data stays on the GPU and everything runs as fast as possible. When using the `cpu` pipeline, simulation can run on either CPU or GPU, depending on the `sim_device` setting, but a copy of the data is always made on the CPU at every step.
* `test=TEST`- If set to `True`, only runs inference on the policy and does not do any training.
* `checkpoint=CHECKPOINT_PATH` - Set to path to the checkpoint to load for training or testing.
* `headless=HEADLESS` - Whether to run in headless mode.
* `experiment=EXPERIMENT` - Sets the name of the experiment.
* `max_iterations=MAX_ITERATIONS` - Sets how many iterations to run for. Reasonable defaults are provided for the provided environments.

Hydra also allows setting variables inside config files directly as command line arguments. As an example, to set the discount rate for a rl_games training run, you can use `train.params.config.gamma=0.999`. Similarly, variables in task configs can also be set. For example, `task.env.enableDebugVis=True`.

#### Hydra Notes

Default values for each of these are found in the `isaacgymenvs/cfg/config.yaml` file.

The way that the `task` and `train` portions of the config works are through the use of config groups. 
You can learn more about how these work [here](https://hydra.cc/docs/tutorials/structured_config/config_groups/)
The actual configs for `task` are in `isaacgymenvs/config/task/<TASK>.yaml` and for train in `isaacgymenvs/config/train/<TASK>PPO.yaml`. 

In some places in the config you will find other variables referenced (for example,
 `num_actors: ${....task.env.numEnvs}`). Each `.` represents going one level up in the config hierarchy.
 This is documented fully [here](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation).

## Tasks

Source code for tasks can be found in `isaacgymenvs/tasks/allegro_hand_baoding_graph.py`. 

Each task subclasses the `VecEnv` base class in `isaacgymenvs/base/vec_task.py`.

Perception models can be found in `isaacgymenvs/encoder`.

## Domain Randomization

IsaacGymEnvs includes a framework for Domain Randomization to improve Sim-to-Real transfer of trained
RL policies. You can read more about it [here](docs/domain_randomization.md).


## Citing

Please cite this work as:
```
@misc{makoviychuk2021isaac,
      title={Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning}, 
      author={Viktor Makoviychuk and Lukasz Wawrzyniak and Yunrong Guo and Michelle Lu and Kier Storey and Miles Macklin and David Hoeller and Nikita Rudin and Arthur Allshire and Ankur Handa and Gavriel State},
      year={2021},
      journal={arXiv preprint arXiv:2108.10470}
}

@article{yang2023tacgnn,
  title={Tacgnn: Learning tactile-based in-hand manipulation with a blind robot using hierarchical graph neural network},
  author={Yang, Linhan and Huang, Bidan and Li, Qingbiao and Tsai, Ya-Yen and Lee, Wang Wei and Song, Chaoyang and Pan, Jia},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={6},
  pages={3605--3612},
  year={2023},
  publisher={IEEE}
}

```

