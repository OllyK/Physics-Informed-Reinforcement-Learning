# Third-Order PIRL Extension

## Summary

This branch extends the original PIRL framework from first-order diffraction to high-order diffraction (m=3).

Main changes:
- Consistent period convention
- Arbitrary diffraction-order extraction
- Physics-consistent RL reward
- Third-order pretrained dataset
- RL hyperparameter tuning

---

## Period Convention

Original:
P = λ / sinθ

Updated:
P = mλ / sinθ

Applied consistently in:
- dataset generation
- pretrained model
- RL reward computation
- electromagnetic solver extraction

---

## Dataset

New dataset:
adj_torch_flip_order3

Generated using:
generate_pirl_dataset_flip_3rd.py

Key modification:
eta = de[center + TARGET_ORDER]

with:
TARGET_ORDER = 3

---

## Fourier Truncation Order

Updated:
FTO = 60

Higher diffraction orders required larger Fourier truncation for stable RCWA convergence.

---

## Solver

Added:
custom_solver.py

This extends the original JLAB solver with arbitrary diffraction-order extraction through:

class JLABCodeMultiOrder

without modifying the installed JLAB package directly.

---

## RL Environment

Modified:
deflector_gym/envs/meent_env.py

Reward now computed using:
TARGET_ORDER = 3

using the same diffraction-order convention as the dataset generation stage.

---

## Pretrained Model

New pretrained checkpoint:
adj_torch_flip_order3_base20000.pth

The original first-order pretrained model is not physically consistent for m=3 optimisation.

---

## RL Hyperparameters

Updated:
lr = 1e-4

initial_epsilon = 1.0
final_epsilon = 0.1
epsilon_timesteps = 150000

These improved convergence for high-order diffraction.

---

## Example Commands

### Scratch RL

python main.py \
  --data_dir run_3rd_scratch

### PIRL RL

python main.py \
  --data_dir run_3rd_pirl \
  --pretrained_ckpt adj_torch_flip_order3_base20000.pth
