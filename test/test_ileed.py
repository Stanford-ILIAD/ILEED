# IMPORTS
import os
import argparse
import numpy as np

from torch import device, save
from torch.cuda import is_available, empty_cache

from stable_baselines3 import PPO
# you can technically import just "import expertrank...", but that gives a warning
from ileed.functions.train import train
from ileed.env.make_env import make_env
from ileed.utils.experts import get_noisy_expert_trajs
from ileed.utils.helpers import evaluate_model
from ileed.utils.models import MLP
from ileed.functions.learn import learn

# some constants for simulation
MODEL_PATH = 'data/expertrank/' 
DEVICE = device('cuda:0' if is_available() else 'cpu')
SEED = 0
env_name = 'MiniGrid-Empty-Random-6x6-v0'
SAVE_PATH = 'test/'+env_name+'/'

env = make_env(env_name = env_name, n_env=8, seed=SEED)
minigrid = env_name.startswith('MiniGrid')
model = train(env, SAVE_PATH, timesteps=1000, n_eval=1, seed=SEED, device=DEVICE, minigrid=minigrid)
print("TRAINING FINISHED SUCCESFULLY")

# Check if we can generate trajectories using this model
obs_space = env.observation_space
act_space = env.action_space
indim = np.prod(obs_space.shape)
outdim = act_space.n
noise_levels = [0.1,0.1,0.1,0.1,1]

# load policy
policy_path = SAVE_PATH+'/model_%d.pth'%0
env = make_env(env_name = env_name, n_env=1, seed=SEED)
expert_states, expert_actions, expert_next_states, expert_eval = get_noisy_expert_trajs(
    env=env, 
    expert_filepath=policy_path, 
    noise_levels=noise_levels, 
    traj_size=100, 
    device=DEVICE)
    
data_tup = (expert_states, expert_actions, expert_next_states)
print("GENERATED TRAJECTORIES SUCCESFULLY")

# Check if we can learn from these trajectories
network_args = {'indim':indim,
                'outdim':outdim,
                'hidden_size':32,
                'embed_dim':2,
                'device':DEVICE,
                'seed':SEED,
                'use_latent':True
                }

omega, networks, best_eval, all_logs, all_evals = learn(
            data_tup, 
            env,
            network_args, 
            no_sigmoid=False, 
            loss_type='irt',
            num_iters = 10,
            n_eval = 10,
            num_restarts = 2)

if not os.path.isdir(SAVE_PATH+env_name):
    os.mkdir(SAVE_PATH+env_name)
print('Results will be saved in:\n',SAVE_PATH+env_name)

print('best_eval: ', best_eval)
print('evals: ', all_evals)
print('logs: ', all_logs)
ARGS = None
data = {'exp_args':ARGS,
        'action_network':networks['action_network'].cpu().state_dict(),
        'state_featurizer_network':networks['state_featurizer_network'].cpu().state_dict(),
        # 'latent_net':networks['latent_transition_network'].cpu().state_dict(),
        'omega':omega.cpu(),
        'noise_levels':noise_levels,
        'expert_eval':expert_eval,
        'best_eval':best_eval,
        'all_evals':all_evals,
        'all_logs':all_logs}


save(data,SAVE_PATH+'/results.pt')
del networks
del omega
del expert_states
del expert_actions
del data_tup
empty_cache()

print("FINSHED IMITATING SUCCESFULLY")