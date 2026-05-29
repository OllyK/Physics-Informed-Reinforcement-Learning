# Standard library
import argparse
import os
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from time import sleep

# Third-party
import numpy as np
import torch
import ray
from ray import air, tune
from ray.tune import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from gym.wrappers import TimeLimit

# Local
import deflector_gym
from deflector_gym.wrappers import BestRecorder, Best2RewardRecorder, ExpandObservation
from model import ShallowUQNet
from utils import StructureWriter, seed_all

DATA_DIR = None
PRETRAINED_CKPT = None
LOG_DIR = None

"""
seeding needs to be taken care when multiple workers are used,
that is, you need to set seed for each worker
"""
seed_all(42)

class Callbacks(DefaultCallbacks):
    """
    logging class for rllib
    the method's name itself stands for the logging timing

    e.g. 
    `if step % train_interval == 0:` is equivalent to `on_learn_on_batch`

    you may feel uncomfortable with this logging procedure, 
    but when distributed training is used, it's inevitable
    """

    def on_create_policy(self, *, policy_id, policy) -> None:
        state_dict = torch.load(
            PRETRAINED_CKPT,
            map_location=torch.device('cpu')
        )
        policy.set_weights(state_dict)

    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        print(algorithm.get_policy().model)
        # seed_all(42)

    def _get_max(self, base_env):
        # retrieve `env.best`, where env is wrapped with BestWrapper to record the best structure
        bests = [e.best for e in base_env.get_sub_environments()]
        best = max(bests, key=itemgetter(0))

        return best[0], best[1]

    def _tb_image(self, structure):
        # transform structure to tensorboard addable image
        img = structure[np.newaxis, np.newaxis, :].repeat(32, axis=1)

        return img

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs) -> None:
        eff, struct = self._get_max(base_env)

        episode.custom_metrics['initial_efficiency'] = eff

    def _j(self, a, b):
        return os.path.join(a, b)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs, ) -> None:
        eff, struct = self._get_max(base_env)
        episode.custom_metrics['max_efficiency'] = eff
        filename = 'w' + str(worker.worker_index) + f'_{eff * 100:.6f}'.replace('.', '-')
        filename = self._j(LOG_DIR, filename)
        np.save(filename, struct)
        
class TwoRewardCallbacks(DefaultCallbacks):
    """
    logging class for rllib
    the method's name itself stands for the logging timing

    e.g. 
    `if step % train_interval == 0:` is equivalent to `on_learn_on_batch`

    you may feel uncomfortable with this logging procedure, 
    but when distributed training is used, it's inevitable
    """

    def on_create_policy(self, *, policy_id, policy) -> None:
        state_dict = torch.load(
            PRETRAINED_CKPT,
            map_location=torch.device('cpu')
        )
        policy.set_weights(state_dict)

    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        print(algorithm.get_policy().model)
        # seed_all(42)

    def _get_max(self, base_env):
        # Now returns the full tuple: (eff, struct, eff_on, eff_off)
        bests = [e.best for e in base_env.get_sub_environments()]
        best = max(bests, key=itemgetter(0))
        return best

    def _tb_image(self, structure):
        # transform structure to tensorboard addable image
        img = structure[np.newaxis, np.newaxis, :].repeat(32, axis=1)

        return img

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs) -> None:
        best = self._get_max(base_env)
        eff = best[0]
        episode.custom_metrics['initial_efficiency'] = eff

    def _j(self, a, b):
        return os.path.join(a, b)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs, ) -> None:
        best = self._get_max(base_env)
        eff, struct, eff_on, eff_off = best[0], best[1], best[2], best[3]
        episode.custom_metrics['max_efficiency'] = eff

        if eff_on is not None and eff_off is not None:
            # Multi-RI: include on/off/margin in the filename and save a dict
            filename = (
                'w' + str(worker.worker_index) +
                f'_on{eff_on*100:.6f}_off{eff_off*100:.6f}_m{eff*100:.6f}'
            ).replace('.', '-')
            data = {'structure': struct, 'eff': float(eff), 'eff_on': float(eff_on), 'eff_off': float(eff_off)}
        else:
            # Single-RI
            filename = 'w' + str(worker.worker_index) + f'_{eff * 100:.6f}'.replace('.', '-')
            data = {'structure': struct, 'eff': float(eff)}

        filename = self._j(LOG_DIR, filename)
        np.save(filename, data, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str, default='run',
        help='absolute path to data directory'
    )
    parser.add_argument(
        '--transfer_ckpt', type=str, default=None,
        help='absolute path to checkpoint file to do transfer learning'
    )
    parser.add_argument(
        '--pretrained_ckpt', type=str,
        default=f'{Path(__file__).resolve().parent}/Pretrained_UNet_1100wl_60deg.pt',
        help='absolute path to checkpoint file of pretrained model'
    )
    parser.add_argument(
        '--wavelength', type=int, default=1100,
        help='wavelength of the incident light'
    )
    parser.add_argument(
        '--wavelength_2', type=int, default=0,
        help='alternative wavelength for the incident light (0 for single wavelength)'
    )
    parser.add_argument(
        '--wavelength_3', type=int, default=0,
        help='alternative wavelength for the incident light (0 for single wavelength)'
    )
    parser.add_argument(
        '--angle', type=int, default=60,
        help='target deflection angle condition'
    )
    parser.add_argument(
        '--thickness', type=int, default=325,
        help='thickness of the pillar'
    )
    parser.add_argument(
        '--train_steps', type=int, default=200000,
        help='number of training steps'
    )
    parser.add_argument(
        '--ri_1', type=float, default=1.45,
        help='refractive index of the pillar material'
    )
    parser.add_argument(
        '--ri_2', type=float, default=0.00,
        help='alternative refractive index for the pillar material'
    )
    parser.add_argument(
        '--reward_mode', type=str, default='shaped',
        choices=['margin','weighted_margin','margin_delta','ratio','log_ratio','shaped','original', 'z-score'],
        help='Reward strategy for MultiRIIndex (ignored if single RI).'
    )
    parser.add_argument(
        '--beta', type=float, default=0.99,
        help='Beta parameter for the reward function.'
    )
    parser.add_argument(
        '--eps', type=float, default=1e-8,
        help='Epsilon parameter for the reward function.'
    )

    args = parser.parse_args()

    DATA_DIR = args.data_dir
    LOG_DIR = f"{args.data_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    PRETRAINED_CKPT = args.pretrained_ckpt

    os.makedirs(LOG_DIR, exist_ok=True)

    def try_start_ray(local_mode):
        # On a shared SLURM node Ray over-detects CPUs (it sees all physical cores, not the
        # cgroup allocation) and pre-starts one idle worker per core. On an NFS-mounted venv
        # that storm of simultaneous imports starves Ray's dashboard/agent process, so it
        # misses its registration deadline -- and the raylet "fate-shares" with the agent and
        # exits, surfacing later as "Unable to register worker with raylet. No such file or
        # directory". Bounding num_cpus to the allocation (fewer competing imports) and giving
        # the agent a generous registration timeout keeps the raylet alive.
        num_cpus = (
            int(os.environ["SLURM_CPUS_PER_TASK"])
            if os.environ.get("SLURM_CPUS_PER_TASK")
            else None
        )
        depth = 0
        while True:
            try:
                print("Trying to start ray.")
                ray.init(
                    local_mode=local_mode,
                    include_dashboard=False,
                    num_cpus=num_cpus,
                    object_store_memory=4_000_000_000,  # bounded so Ray can't size off the node's full RAM
                    _system_config={"agent_register_timeout_ms": 600000},  # 10 min; agent needs minutes on NFS
                )
                break
            except Exception as e:
                waittime = np.random.randint(1, 10 * 2**depth)
                print(f"Failed to start ray on attempt {depth+1} ({e!r}). Retrying in {waittime} seconds...")
                sleep(waittime)
                depth += 1

    try_start_ray(local_mode=False)
    
    # TODO: This should be tidied up into classes
    # decide which environment to use
    # single or two reward mode
    two_reward_mode_fl = args.ri_2 != 0.0
    two_wavelength_mode_fl = args.wavelength_2 != 0 and args.wavelength_3 != 0
    if two_wavelength_mode_fl:
        assert not two_reward_mode_fl, "Multi-wavelength mode only works without multi-RI environment"
        print("Using MultiWavelengthIndex environment")
        env_id = 'MultiWavelengthIndex-v0'
        env_config = {
            'wavelength': args.wavelength,
            'wavelength_off1': args.wavelength_2,
            'wavelength_off2': args.wavelength_3,
            'desired_angle': args.angle,
            'thickness': args.thickness,
            'refractive_index': args.ri_1,
        }
        cbs = TwoRewardCallbacks
        best_recorder = Best2RewardRecorder

    elif two_reward_mode_fl:
        env_id = 'MultiRIIndex-v0'
        env_config = {
            'wavelength': args.wavelength,
            'desired_angle': args.angle,
            'thickness': args.thickness,
            'refractive_index': args.ri_1,
            'refractive_index_2': args.ri_2,
            'reward_mode': args.reward_mode,
            'beta': args.beta,
            'eps': args.eps,
        }
        cbs = TwoRewardCallbacks
        best_recorder = Best2RewardRecorder
    else:
        env_id = 'MeentIndex-v0'
        env_config = {
            'wavelength': args.wavelength,
            'desired_angle': args.angle,
            'thickness': args.thickness,
            'refractive_index': args.ri_1,
        }
        cbs = Callbacks
        best_recorder = BestRecorder

    model_cls = ShallowUQNet  # model_cls = ShallowUQNet / FCNQNet / FCNQNet_heavy

    def make_env(config):
        env = deflector_gym.make(env_id, **config)
        env = best_recorder(env)
        env = ExpandObservation(env)
        env = StructureWriter(env, DATA_DIR)
        env = TimeLimit(env, max_episode_steps=128)

        return env

    register_env(env_id, lambda c: make_env(env_config))
    ModelCatalog.register_custom_model(model_cls.__name__, model_cls)

    from configs.simple_q import single_worker as config

    config.framework(
        framework='torch'
    ).environment(
        env=env_id,
        env_config=env_config,
        normalize_actions=False
    ).callbacks(
        cbs  # register logging
    ).training(
        model={'custom_model': model_cls}
    ).debugging(
        # seed=tune.grid_search([1, 2, 3, 4, 5]) # if you want to run experiments with multiple seeds
    )

    algo = config.build()
    if args.transfer_ckpt:
        algo.load_checkpoint(args.transfer_ckpt)
    stop = {
        "timesteps_total": args.train_steps,
    }
    tuner = tune.Tuner(
        'SimpleQ',
        param_space=config.to_dict(),
        # tune_config=tune.TuneConfig(), # for hparam search
        run_config=air.RunConfig(
            stop=stop,
            local_dir=DATA_DIR,
            name=LOG_DIR,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute='episode_reward_max',
                checkpoint_score_order='max',
                checkpoint_frequency=1,
                checkpoint_at_end=True,
            ),
        ),
    )

    results = tuner.fit()
    ray.shutdown()
