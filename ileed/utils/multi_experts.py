'''
This file contains the main classes required for running minigrid 
expertiments and analysis. This is similair to the experts.py file, 
but we are rewriting the functios to be compatible with generating our 
own "state dependent" noise. 
'''
#------------------------------------------------------------------------------------#
# IMPORTS
#------------------------------------------------------------------------------------#
from numpy import zeros
from numpy.random import uniform, choice
#------------------------------------------------------------------------------------#
class multi_experts(object):
    '''
    This will just be a class that iterates which expert to use based on env_id 
    We will assume that the experts used are PPO class experts from sb3
    Parameters:
    exp_list - list of sb3.PPO class models that are the experts we will use
    omega - omega[i]=p--> the agent chooses best action a* with probability p, 
            and acts randomly with probability (1-p)
    n     - number of actions that the expert can sample from

    '''
    def __init__(self, exp_list, omega, n):
        super(multi_experts, self).__init__()
        self.exp_list = exp_list
        self.omega = omega
        self.n = n

    def predict(self, obs, env_id):
        '''
        returns the action recommendation based on obs and env_id
        the state dependent noise is added here based on env_id and omega
        '''
        if uniform() < self.omega[env_id]:
            return self.exp_list[env_id].policy.predict(obs)
        else:
            return choice(self.n), None
    
    def collect_trajectories(self, env, length):
        '''
        Calculates the trajectory of our experts within the multi_env given
        NOTE: env should be of class multi_env defined in expertrank/env/multi_env.py

        Returns tupleof S, A, R, nextS, env_id arrays (python lists)
        '''
        S, A, R, nextS, env_id = [], [], [], [], []
        # this will exit when we got the desired traj
        while len(S) < length:
            obs = env.reset()
            done = False
            while not done:
                # on policy 
                act, _ = self.predict(obs, env.env_id)
                obs_new, rew, done, _ = env.step(act)
                # store the corresponding trajectory
                S.append(obs)
                A.append(act)
                R.append(rew)
                nextS.append(obs_new)
                env_id.append(env.env_id)
                # on to the next step
                obs = obs_new

        return S, A, R, nextS, env_id        

    def evaluate(self, env, n_eval_episodes):
        '''
        Evaluates over n_eval episodes of the env given

        returns btoh mean and std of rewards collected
        '''
        rew = zeros(n_eval_episodes)
        for i_eps in range(n_eval_episodes):
            obs = env.reset()
            done = False
            while not done:
                act, _ = self.predict(obs, env.env_id)
                obs_new, r, done, _ = env.step(act)
                rew[i_eps]+=r
                obs = obs_new
        return rew.mean(), rew.std()

