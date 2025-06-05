# Physics-Informed Metasurface Design

This repository contains an implementation of Physics-Informed Reinforcement Learning for optimising optical metasurface designs. The system uses Ray RLlib with a custom environment (DeflectorGym) to simulate and optimise the efficiency of beam deflectors.

## Overview

This project enables the design of metasurfaces with high deflection efficiency by using pre-trained physics models and reinforcement learning. It combines:

- A custom Gymnasium environment for RCWA (Rigorous Coupled-Wave Analysis) simulation
- Physics-informed neural networks to accelerate the design process
- Ray RLlib for distributed reinforcement learning training

## Installation

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
