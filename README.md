# Physics-Informed Metasurface Design

This repository contains an implementation of Physics-Informed Reinforcement Learning for optimising optical metasurface designs. The system uses Ray RLlib with a custom environment (DeflectorGym) to simulate and optimise the efficiency of beam deflectors.

## Overview

This project enables the design of metasurfaces with high deflection efficiency by using pre-trained physics models and reinforcement learning. It combines:

- A custom Gymnasium environment for RCWA (Rigorous Coupled-Wave Analysis) simulation
- Physics-informed neural networks to accelerate the design process
- Ray RLlib for distributed reinforcement learning training


## Setup on ARC HPC system

There are a number of steps to follow in order to set this up on ARC. 

1. Login to the HTC login node: `ssh <username>@htc-login.arc.ox.ac.uk`
2. Request an interactive session: `srun -p interactive --pty /bin/bash`
3. Clone this repository to your `$DATA` directory: 
```shell
mkdir -p $DATA/repos
cd $DATA/repos
git clone --recursive git@github.com:OllyK/Physics-Informed-Reinforcement-Learning.git
cd Physics-Informed-Reinforcement-Learning
```
4. Run the installation script to create a conda environment called `nano_rl` at `$DATA/venvs/nano_rl` that has all the necessary packages installed. 
```shell
bash installation.sh
```
5. Once the script has run, you can then exit the interactive session to go back to the login node:
```shell
exit
```

## Running on ARC

A script is included in this repository for submitting a job to the HTC cluster on ARC. To use it do the following:

1. Follow the instructions [to setup your environment on ARC](#setup-on-arc-hpc-system)
2. Login to the HTC login node if not already done
3. Ensure you have edited the script `run_nano_rl_h100.sh` in this repository (`cd $DATA/repos/Physics-Informed-Reinforcement-Learning`) to replace `<insert your email here>` with your email address.
4. Ensure there is a directory named `nano_rl_logs` in your current working directory (slurm output logs will be saved here) `mkdir -p nano_rl_logs`. 
5. Submit the script to the Slurm scheduler
```shell
sbatch run_nano_rl_h100.sh
```
6. If the script runs successfully data will be output to a subdirectory in the `Physics-Informed-Reinforcement-Learning/runs` directory. Details of which subdirectory can be found in the  Slurm output log for the job (`nano_rl_logs/nano_rl_xxxxxx.out`)


## Local Installation

### Clone the Repository
```shell
git clone https://github.com/jLabKAIST/physics-informed-metasurface.git
cd physics-informed-metasurface
```

### Requirements

```shell
pip install -r requirements.txt
```
### DeflectorGym
`DeflectorGym` is our custom environment for RCWA simulation.
```shell
cd deflector-gym
pip install -e .
```

## Usage

### Basic Training

To start training a model with default parameters:

```shell
python main.py
```

### Training with Custom Parameters

You can customise the training process by specifying various parameters:

```shell
python main.py --data_dir run_experiment --wavelength 1100 --angle 60 --thickness 325 --train_steps 200000
```

### Configuration Options

The following parameters can be adjusted:
- `--data_dir`: Directory to store results
- `--wavelength`: Wavelength of the incident light (in nm)
- `--angle`: Target deflection angle (in degrees)
- `--thickness`: Thickness of the metasurface pillars (in nm)
- `--train_steps`: Number of training steps
- `--pretrained_ckpt`: Path to a pretrained model checkpoint for transfer learning
- `--transfer_ckpt`: Path to a checkpoint for continuing previous training

### Model Types

This repository implements several neural network architectures for metasurface design:

1. **ShallowUQNet**: A U-Net style architecture with skip connections for detailed spatial features
2. **FCNQNet**: A fully connected network for simpler designs
3. **FCNQNet_heavy**: A larger fully connected network with higher capacity

To select a model, modify the `model_cls` variable in `main.py`:
```python
model_cls = ShallowUQNet  # Options: ShallowUQNet / FCNQNet / FCNQNet_heavy
```

### Distributed Training

The system supports distributed training using Ray. There are two configurations available in `configs/simple_q.py`:
- `single_worker`: For single-machine training
- `multiple_worker`: For distributed training with multiple workers

To use distributed training:
```shell
python main.py --cpus 16
```

### Visualising Results

Results are automatically saved in the specified data directory. Each training run creates a timestamped folder containing:
- Best metasurface structures as NumPy files
- Training metrics
- Checkpoints of trained models

## Advanced Usage

### Transfer Learning

To continue training from a previously trained model:

```shell
python main.py --transfer_ckpt path/to/checkpoint --wavelength 1300 --angle 45
```

### Using Pre-trained Physics Models

The system leverages pre-trained physics models to accelerate training:

```shell
python main.py --pretrained_ckpt pretrained/custom_model.pt
```

### Custom Environments

The training environment can be customised by modifying the wrappers in `main.py`. The default setup includes:
- `BestRecorder`: Tracks the best structure found during training
- `ExpandObservation`: Enhances the observation space
- `StructureWriter`: Saves structures to disk
- `TimeLimit`: Limits episode length
