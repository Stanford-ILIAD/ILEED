from stable_baselines3.common.env_util import make_vec_env
from ileed.env.mdp_gym import mdp_w_exp
from gym_minigrid.wrappers import FlatObsWrapper, ImgObsWrapper, RGBImgObsWrapper

def make_env(env_name: str, n_env: int, seed=0):
    '''
    This will return a vectorized environment to be used in the algorithm.
    The reason for the custom function is to have env_type=='apples' 
    correspond to our custom environment. Otherwise, this relies on regular gym 
    function. 
    '''
    if (env_name =='apples') or (env_name =='bombs') or (env_name =='both'):
        env_args = {
            'T':100,
            'seed': seed,
            'dim':10, 
            'v_dim':7, 
            'R':1, 
            'n_goals':[10,50], 
            'obs_method':'centergrid',
        }

        if env_name == 'apples':
            env_args['start_phase'] = 1
            env_args['T_p'] = env_args['T']
        elif env_name == 'bombs':
            env_args['start_phase'] = -1
            env_args['T_p'] = env_args['T']
        elif env_name == 'both':
            env_args['start_phase'] = 1
            env_args['T_p'] = 20
        env = make_vec_env(mdp_w_exp, n_env, seed, env_kwargs=env_args)        
    
    elif (env_name.startswith('MiniGrid')):
        env=make_vec_env(env_name,n_envs=n_env,wrapper_class=FlatObsWrapper, seed=seed)
        # env=make_vec_env(env_name,n_envs=n_env, wrapper_class=RGBImgObsWrapper)
    else:
        env = make_vec_env(env_name, n_env, seed)
    return env