# IMPORTS
import os
import argparse
import torch
import numpy as np

from ileed.env.make_env import make_env
from ileed.functions.train import train

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

all_env_names = ['MiniGrid-Empty-Random-6x6-v0',
                'MiniGrid-Dynamic-Obstacles-Random-6x6-v0',
                'MiniGrid-LavaGapS5-v0',
                'MiniGrid-Unlock-v0']

def main():
    for env_name in all_env_names:
        env = make_env(env_name, n_env=8, seed=ARGS.seed)
        save_path = 'data/ileed/%s'%(env_name)
        # model is saved inside train() function call
        minigrid = env_name.startswith('MiniGrid')
        model = train(env, save_path, timesteps=200000, n_eval=2, seed=ARGS.seed, device=DEVICE, minigrid=minigrid)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # these three args are irrelevant for now
    parser.add_argument('--seed', type=int, default=0, help="embedding dimension")
    ARGS = parser.parse_args()
    print(ARGS)
    
    main()