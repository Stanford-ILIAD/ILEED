# IMPORTS
import os
import argparse
import pdb
import torch
import numpy as np
import pickle 
import random

from ileed.env.make_env import make_env
from ileed.utils.experts import get_noisy_expert_trajs
from ileed.utils.helpers import evaluate_model
from ileed.utils.models import MLP
from ileed.functions.learn import learn

from functions.algorithms import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

all_noise_levels = [np.array([0.99,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]),
                    np.array([0.99,0.99,0.99,0.99,0.99,0.01,0.01,0.01,0.01,0.01]),
                    np.array([0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99]),
                    np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])]

all_env_names = ['MiniGrid-Empty-Random-6x6-v0',
                'MiniGrid-Dynamic-Obstacles-Random-6x6-v0',
                'MiniGrid-LavaGapS5-v0',
                'MiniGrid-Unlock-v0']

def main():
    for seed in range(2):
        torch.manual_seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        result_dict = dict()
        for env_name in all_env_names:
            # store results and expert evalutations for each noise
            result = np.zeros((len(all_noise_levels),3,2)) #(noise_distrib,algo,[mean,std])
            all_evals = []
            # load env
            env = make_env(env_name,n_env=1,seed=seed)
            # iterate over noise levels 
            for i_noise in range(len(all_noise_levels)):
                # collect trajectories associated with this noise level
                expert_states, expert_actions, expert_next_states, expert_eval = get_noisy_expert_trajs(
                    env=env, 
                    expert_filepath='data/expertrank/%s/0.pth'%(env_name), 
                    noise_levels=all_noise_levels[i_noise], 
                    traj_size=ARGS.traj_size, 
                    device=DEVICE)
                all_evals.append(expert_eval)
                data_tup = (expert_states, expert_actions, expert_next_states)
                # pickle.dump(data_tup,open('temp_%d.p'%seed,'wb'))
                # RUN ALL 3 Algorithms
                result[i_noise, 0] = run_bc(data_tup, env, ARGS.hidden_size, ARGS.embed_dim, DEVICE, seed)
                result[i_noise, 1] = run_mle(data_tup, env, ARGS.hidden_size, ARGS.embed_dim, DEVICE, seed)
                result[i_noise, 2] = run_mle_state(data_tup, env, ARGS.hidden_size, ARGS.embed_dim, DEVICE, seed)

                torch.cuda.empty_cache()
                tf.keras.backend.clear_session()
            # print(result)
            result_dict[env_name] = {'result': result, 'expert_evals': all_evals}
        print(result_dict)
        pickle.dump(result_dict,open('data/%s/result_%d.p'%(ARGS.save_dir,seed),'wb'))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # these three args are irrelevant for now
    parser.add_argument('--embed_dim', type=int, default=2, help="embedding dimension")
    parser.add_argument('--seed', type=int, default=0, help="embedding dimension")
    parser.add_argument('--hidden_size', type=int, default=4, help="dimension of hidden layer in MLPs")
    parser.add_argument('--traj_size', type=int, default=1000, help="length of traj to load from EACH expert")
    parser.add_argument('--save_dir', type=str, default='ileed', help="exp dir")
    ARGS = parser.parse_args()
    print(ARGS)
    
    main()
