'''
This file contains helper functions to make it possible to interact 
with the learned expert policies. The policies are generally stored
as PPO class (default MlpPolicy) from sb3. 
'''
#------------------------------------------------------------------------------------#
# IMPORTS
#------------------------------------------------------------------------------------#
from types import MethodType
from typing import Optional 
import pdb

from torch import Tensor, stack
from numpy.random import uniform

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
#    make_proba_distribution,
)
#------------------------------------------------------------------------------------#
# PART I: Noise functions
#------------------------------------------------------------------------------------#     
def model_add_uniform_noise(expert_model, r):
    '''
        expert acts based on p(a|s), r of the time
        expert acts randomly, (1-r) of the time
        r=1 keeps the original expert
        r=0 makes expert completely random
    '''
    
    def _get_action_dist_from_latent(self, latent_pi: Tensor, latent_sde: Optional[Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        use_unif = uniform() > r
        
        if isinstance(self.action_dist, DiagGaussianDistribution):
            std = self.log_std
            if use_unif: std /= max(r,1e-5)
            return self.action_dist.proba_distribution(mean_actions, std)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            std = self.log_std
            if use_unif: std /= max(r,1e-5)
            return self.action_dist.proba_distribution(mean_actions, std, latent_sde)

        if use_unif: mean_actions *= 0
        
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")
    
    expert_model.policy._get_action_dist_from_latent = MethodType( _get_action_dist_from_latent, expert_model.policy )
    return expert_model
#------------------------------------------------------------------------------------#
# PART II: Collecting Trajectories
#------------------------------------------------------------------------------------#
def collect_trajectories(model, env, length, deterministic=False):
    '''
    Calculates the trajectory of our expert within some env

    Inputs:
        env           - env class to evaluate on
        deterministic - BOOL: wether the policy we use is deterministic
        length        - int: how many timesteps we want the collected traj to be

    Returns tuple of S A R arrays
    '''
    S, A, R, nextS = [], [], [], []

    # this will exit when we got the desired traj
    while len(S) < length:
        obs = env.reset()
        done = False
        while not done:
            # on policy 
            act, _ = model.policy.predict(obs, deterministic=deterministic)
            obs_new, rew, done, _ = env.step(act)
            # flatten for easier storing
            obs = obs.reshape(-1)
            # store the corresponding trajectory
            S.append(obs)
            A.append(act)
            R.append(rew)
            nextS.append(obs_new.reshape(-1))
            # on to the next step
            obs = obs_new

    return S, A, R, nextS
#------------------------------------------------------------------------------------#
# PART III: Collecting NOISY Trajectories
#------------------------------------------------------------------------------------#
def get_noisy_expert_trajs(env, expert_filepath, noise_levels, traj_size, device):
    '''
    This function takes an individual expert policy, and generates
    expert trajectories using the defined noise_levels. 
    Inputs:
        env              - env class to evaluate 
        expert_filepath  - filepath of the expert policy (sb3 PPO)
        noise_levels     - list of noise_levels to be used, 
                           NOTE: this is only used if collect_new_traj is True
        traj_size        - size of trajectory 
        device           - id of device

    Outputs:
        Torch tensors for States, Actions, and evaluation dict

    '''
    eval = {}
    # Load experts using sb3 PPO class
    # pdb.set_trace()
    expert = PPO.load(expert_filepath, custom_objects= {
      "learning_rate": 0.0,
      "lr_schedule": lambda _: 0.0,
      "clip_range": lambda _: 0.0,
  })
    # confirm the mean reward (std should be small)
    mean_reward, std_reward = evaluate_policy(model=expert, env=env, n_eval_episodes=100)
    print("expert reward avg: %.2f std: %.2f" % (mean_reward, std_reward))
    eval['expert']=[mean_reward,std_reward]
    # initialize variables to store the trajectories
    expert_states = [None for _ in noise_levels]
    expert_actions = [None for _ in noise_levels]
    expert_next_states = [None for _ in noise_levels]
    # go through each noise level defined
    for i, expert_noise in enumerate(noise_levels):
        # this returns a PPO class expert with uniform noise added to their actions
        # as of now, 0\leq r\leq 1 is the portion of actions taken uniform random
        noise_expert = model_add_uniform_noise(expert, r = expert_noise)
        # evaluate the base expert
        mean_reward, std_reward = evaluate_policy(model=noise_expert, env=env)
        print("expert noise %.2f reward avg: %.2f std: %.2f" % (expert_noise, mean_reward, std_reward))
        eval['noise_%d'%i]=[mean_reward,std_reward]
        # collect trajectories
        trajS, trajA, _, trajnextS = collect_trajectories(noise_expert, env, length=traj_size)
        expert_states[i] = Tensor(trajS[0:traj_size]).float()
        expert_actions[i] = Tensor(trajA[0:traj_size]).long()
        expert_next_states[i] = Tensor(trajnextS[0:traj_size]).float()
    return stack(expert_states).to(device), stack(expert_actions).to(device), stack(expert_next_states).to(device), eval

def get_expert_trajs(env, exp_list, traj_size, device):
    '''
    This function takes an individual expert policy, and generates
    expert trajectories using the defined noise_levels. 
    Inputs:
        env              - env class to evaluate 
        expert_filepath  - filepath of the expert policy (sb3 PPO)
        noise_levels     - list of noise_levels to be used, 
                           NOTE: this is only used if collect_new_traj is True
        traj_size        - size of trajectory 
        device           - id of device

    Outputs:
        Torch tensors for States, Actions, and evaluation dict

    '''
    eval = {}
    # initialize variables to store the trajectories
    expert_states = [None for _ in exp_list]
    expert_actions = [None for _ in exp_list]
    expert_next_states = [None for _ in exp_list]
    # go through each noise level defined
    for i, expert in enumerate(exp_list):
        mean_reward, std_reward = evaluate_policy(model=expert, env=env, n_eval_episodes=100)
        print("expert num %d reward avg: %.2f std: %.2f" % (i, mean_reward, std_reward))
        eval['exp_%d'%i]=[mean_reward,std_reward]
        # collect trajectories
        trajS, trajA, _, trajnextS = collect_trajectories(expert, env, length=traj_size)
        expert_states[i] = Tensor(trajS[0:traj_size]).float()
        expert_actions[i] = Tensor(trajA[0:traj_size]).long()
        expert_next_states[i] = Tensor(trajnextS[0:traj_size]).float()
    return stack(expert_states).to(device), stack(expert_actions).unsqueeze(dim=-1).to(device), stack(expert_next_states).to(device), eval
