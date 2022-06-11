#------------------------------------------------------------------------------------#
# IMPORTS
#------------------------------------------------------------------------------------#
from stable_baselines3 import PPO
# relative
from ileed.utils.helpers import VideoRecorderCallback
from stable_baselines3.common.callbacks import EvalCallback
#------------------------------------------------------------------------------------#
# Main train Function
#------------------------------------------------------------------------------------#
def train(env, save_path, timesteps=200000, n_eval=20, seed=0, device='cuda:0', minigrid=False):
    '''
    Main function for training experts from scratch. This will utilize 
    SB3 PPO function by default, relying on MlpPolicy
    TODO: Maybe have more policies possible for training

    Inputs:
        env         - env class, either any gym or one from env/mdp_gym.py
        save_path   - path of parent folder where the results will be saved
        timesteps   - # of timesteps to train for
        seed       
        device
        minigrid    - if True, uses parameters that work better for minigrid env
    
    Outputs:
        returns the model used for training directly 
    '''

    if minigrid:
        model = PPO('MlpPolicy', 
                    env,
                    verbose=0,
                    seed=seed,
                    device=device,
                    n_steps=128,
                    batch_size=4,
                    learning_rate=2.5e-4)
    else:
        model = PPO('MlpPolicy', 
                    env,
                    verbose=0,
                    seed=seed,
                    device=device)
    # assumes 8 env created
    eval_callback = EvalCallback(env, 
                                best_model_save_path=save_path,
                                log_path=save_path, 
                                eval_freq=timesteps//(n_eval*8),
                                n_eval_episodes=100,
                                deterministic=True, 
                                render=False)

    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(save_path+'model_'+str(seed)+'.pth')
    return model
