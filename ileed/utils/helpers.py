'''
This contains helper functions used throughout the utils folder.
This way, each file in utils/ only has one main function, and relies on 
the helpers inside here.
'''
#------------------------------------------------------------------------------------#
# IMPORTS
#------------------------------------------------------------------------------------#
from typing import Any, Dict
from torch import cat, gather, log, logsumexp, ByteTensor
from torch.nn import SmoothL1Loss

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
#------------------------------------------------------------------------------------#
# PART I: VideoRecorder to be used when training new experts
#------------------------------------------------------------------------------------#
class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, 
                recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(ByteTensor([screens]), fps=3),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True
#------------------------------------------------------------------------------------#
# PART II: losses used in loss.py (for main algorithm in learn.py)
#------------------------------------------------------------------------------------#
def IRT_logloss(sigma, actions, prob_a_vec):
    '''
    This loss is based on the IRT model, as of now assuming worst case scenario 
    is random actions.

    Inputs:
        sigma
        actions
        prob_a_vec
    
    Outputs:
        log likelihood of loss 
    '''
    # sigma[i][j] = probability of expert i making the correct action at states[i][j]
    # print(actions)
    # breakpoint()
    prob_a = gather(input=prob_a_vec, dim=2, index=actions) 
    # prob_a[i][j] = prob of action taken by expert i being optimal at states[i][j]
    prob_a = prob_a.squeeze(-1)
    log_likelihood = log(sigma*prob_a + (1-sigma)*(1-prob_a)/(prob_a_vec.size(-1)-1))
    # log_likelihood = prob_a*log(sigma) + (1-prob_a)/(prob_a_vec.size(-1)-1)*log((1-sigma))


    return -1*log_likelihood.sum()

def noisy_rational_logloss(sigma, actions, prob_a_vec):
    '''
    This loss is based on the ...
    
    Inputs:
        sigma
        actions
        prob_a_vec
    
    Outputs:
        log likelihood of loss 
    '''
    # sigma corresponds to noisy rationality (1 being rational, 0 being irrational)
    temp = sigma.unsqueeze(-1)
    log_a_vec = log(prob_a_vec)
    log_a_vec = log_a_vec*temp - logsumexp(log_a_vec*temp, dim=-1, keepdim=True)
    return gather(input=log_a_vec, dim=2, index=actions).sum()
#------------------------------------------------------------------------------------#
# PART III: loss and evaluation function used in learn.py
#------------------------------------------------------------------------------------#
def evaluate_model(action_network, env, n_eval_episodes):
    '''
    Evaluates the action network using MlpPolicy class in sb3 PPO
    sets deterministic = False
    
    Inputs:
        action_network
        env
        n_eval_episodes
    
    Outputs:
        tuple of mean and std of episodic reward (calc over n_eval_episodes)
    '''
    def predict(observation, deterministic=False):
        obs = observation.reshape(observation.size(0), -1)
        out = action_network.forward(obs.float())
        return out.argmax(dim=-1)
    
    model = PPO("MlpPolicy", env, device=action_network.device)
    model.policy._predict = predict
    
    mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=n_eval_episodes)
    return mean_reward, std_reward

def state_featurizer_loss(data_tup, state_featurizer_network, latent_transition_network):
    '''
    Loss for the state_featurizer_network (predicts the transitions)
    
    Inputs:
        data_tup
        state_featurizer_network
        latent_transition_network

    
    Outputs:
        Smooth_l1 loss over the error in 
        transitions predicted over the latent space
    '''    
    states, actions, next_states = data_tup
    
    loss_func = SmoothL1Loss()
    
    states_features = state_featurizer_network(states)
    next_states_features = state_featurizer_network(next_states)
    predicted_next_states_features = latent_transition_network( cat([states_features, actions], dim=-1) )

    # print( next_states_features, predicted_next_states_features )
    loss = loss_func( next_states_features, predicted_next_states_features )
    return loss
#------------------------------------------------------------------------------------#
# losses used in loss.py (for main algorithm in learn.py)
#------------------------------------------------------------------------------------#