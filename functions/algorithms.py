# This file contains all algorithm functions we are running in a way that they return
# a numpy array of [mean_reward, std_reward]

# IMPORTS
import gym
import joblib
import numpy as np
import sys
import os
import tensorflow as tf
import pdb

# expertrank
from ileed.functions.learn import learn
from ileed.utils.helpers import evaluate_model
# rllab

def run_bc(data_tup, env, hidden_size, embed_dim, device, seed):
    '''
    Runs the Behavioural Cloning Algorithm using Cross Entropy Loss
    Inputs:
        env         - Gym environment class
        hidden_size - integer for dimension of the one hidden layer
        embed_dim   - dimension to embed the skill vector omega
    Outputs:
        numpy.array containing mean and std of reward evaluated over 1000 episodes
    '''
    # load env params
    obs_space = env.observation_space
    act_space = env.action_space
    indim = np.prod(obs_space.shape)
    outdim = act_space.n
    # setup exp
    network_args = {'indim':indim,
                    'outdim':outdim,
                    'hidden_size':hidden_size,
                    'embed_dim':embed_dim,
                    'device':device,
                    'seed':seed,
                    'use_latent':False
                    }
    # run
    _, networks, _, _, _ = learn(
                data_tup, 
                env,
                network_args, 
                no_sigmoid=False, 
                loss_type='bc',
                num_iters = 2000,
                n_eval = 1,
                num_restarts = 20)
    # eval
    mean_reward, std_reward = evaluate_model(networks['action_network'], env, n_eval_episodes=1000)
    print("learned reward avg: %.2f std: %.2f" % (mean_reward, std_reward))
    return np.array([mean_reward,std_reward])

def run_mle(data_tup, env, hidden_size, embed_dim, device, seed):
    '''
    Runs our Annotation Algorithm
    Inputs:
        env         - Gym environment class
        hidden_size - integer for dimension of the one hidden layer
        embed_dim   - dimension to embed the skill vector omega
    Outputs:
        numpy.array containing mean and std of reward evaluated over 1000 episodes
    '''
    # load env params
    obs_space = env.observation_space
    act_space = env.action_space
    indim = np.prod(obs_space.shape)
    outdim = act_space.n
    # setup exp
    network_args = {'indim':indim,
                    'outdim':outdim,
                    'hidden_size':hidden_size,
                    'embed_dim':embed_dim,
                    'device':device,
                    'seed':seed,
                    'use_latent':False
                    }
    # run
    _, networks, _, _, _ = learn(
                data_tup, 
                env,
                network_args, 
                no_sigmoid=False, 
                loss_type='irt',
                num_iters = 2000,
                n_eval = 1,
                num_restarts = 20)
    # eval
    mean_reward, std_reward = evaluate_model(networks['action_network'], env, n_eval_episodes=1000)
    print("learned reward avg: %.2f std: %.2f" % (mean_reward, std_reward))
    return np.array([mean_reward,std_reward])

def run_mle_state(data_tup, env, hidden_size, embed_dim, device, seed):
    '''
    Runs our Annotation Algorithm WITH the transient network
    Inputs:
        env         - Gym environment class
        hidden_size - integer for dimension of the one hidden layer
        embed_dim   - dimension to embed the skill vector omega
    Outputs:
        numpy.array containing mean and std of reward evaluated over 1000 episodes
    '''
    # load env params
    obs_space = env.observation_space
    act_space = env.action_space
    indim = np.prod(obs_space.shape)
    outdim = act_space.n
    # setup exp
    network_args = {'indim':indim,
                    'outdim':outdim,
                    'hidden_size':hidden_size,
                    'embed_dim':embed_dim,
                    'device':device,
                    'seed':seed,
                    'use_latent':True
                    }
    # run
    _, networks, _, _, _ = learn(
                data_tup, 
                env,
                network_args, 
                no_sigmoid=False, 
                loss_type='irt',
                num_iters = 2000,
                n_eval = 1,
                num_restarts = 20)
    # eval
    mean_reward, std_reward = evaluate_model(networks['action_network'], env, n_eval_episodes=1000)
    print("learned reward avg: %.2f std: %.2f" % (mean_reward, std_reward))
    return np.array([mean_reward,std_reward])
