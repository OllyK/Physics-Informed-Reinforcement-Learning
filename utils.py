import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import gym
import torch

MB = 1024 * 1024
GB = MB * 1024


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class OneHot(gym.ObservationWrapper):
    def __init__(self, env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.observation_space = gym.spaces.Box(
            low=0., high=1.,
            shape=(256,),  #### TODO fix shape
            dtype=np.float64
        )

    def observation(self, obs):
        obs[obs == -1] = 0

        return obs


class StructureWriter(gym.Wrapper):
    def __init__(
            self,
            env,
            data_dir,
            max_folder_size=100 * GB,  # 100GB
            **kwargs
    ) -> None:
        super().__init__(env, **kwargs)
        self.data_dir = self._j(data_dir, env.unwrapped.__class__.__name__)

        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        folder_size = sum(
            file.stat().st_size for file in Path(self.data_dir).rglob('*'))
        print(
            f"""writing files to {self.data_dir} current folder size: {folder_size / GB:.3f}GB"""
        )
        self.disabled = False

        if folder_size > max_folder_size:
            self.disabled = True
            print(
                f"""folder size is too large: {folder_size / GB}GB > {max_folder_size / GB}GB ignoring {self.__class__}"""
            )

    def _j(self, a, b):
        return os.path.join(a, b)

    def reset(self, **kwargs):
        """
        when episode is done,
        the structure(metasurface) is written to the file

        For single-index envs: filename encodes eff.
        For multi-RI envs: filename encodes eff_on, eff_off, margin.
        Saved file now contains a dict:
          {
            'structure': <1D array>,
            'eff': <margin or single eff>,
            'eff_on': <present only for MultiRIIndex>,
            'eff_off': <present only for MultiRIIndex>
          }
        """
        obs = self.env.reset(**kwargs)

        if not self.disabled:
            unwrapped = self.env.unwrapped
            # detect multi-RI
            has_multi = hasattr(unwrapped, 'eff_on') and hasattr(unwrapped, 'eff_off')
            if has_multi:
                eff_on = float(unwrapped.eff_on)
                eff_off = float(unwrapped.eff_off)
                margin = float(unwrapped.eff)  # already eff_on - eff_off
                filename = (
                    f'on{eff_on*100:.6f}_off{eff_off*100:.6f}_m{margin*100:.6f}'
                    .replace('.', '-')
                )
            else:
                eff_single = float(getattr(unwrapped, 'eff', 0.0))
                filename = f'{eff_single * 100:.6f}'.replace('.', '-')

            filename += '_' + datetime.now().strftime('%Y%m%d-%H%M%S')
            filename = self._j(self.data_dir, filename)

            data = {
                'structure': obs[0],  # original structure
                'eff': float(getattr(unwrapped, 'eff', 0.0)),
            }
            if has_multi:
                data['eff_on'] = eff_on
                data['eff_off'] = eff_off

            # store dictionary (np.save will pickle automatically)
            np.save(filename, data, allow_pickle=True)

        return obs
