import gym

class multi_env(gym.Env):
    """Custom environment that follows gym interface:
    this env combines the list of GYM envs given such that 
    trajectories generated follow the list of envs until the 
    last environment returns Done."""
    metadata = {'render.modes': ['human']}
    def __init__(self, env_list):
        super(multi_env, self).__init__()
        # ALL ENV MUST HAVE SAME ACT/OBS space and by gym env themselves
        self.env_list = env_list
        self.action_space = env_list[0].action_space
        self.observation_space = env_list[0].observation_space
        # this will keep track of the current env used
        self.env_id = 0
        self.n_envs = len(env_list)
    def step(self, action):
        '''we step the current env in list, unless it is done
        if done, we incrememnt env_id, unless it is the last one'''
        # breakpoint
        observation, reward, done, info = self.env_list[self.env_id].step(action)
        if done:
            # print('inside: ',self.env_id,reward)
            # check if not last (will return done=False and new observation)
            if not self.env_id==self.n_envs-1:
                self.env_id += 1
                observation = self.env_list[self.env_id].reset()
                done = False
        # if the last env triggers done, it will be sent as True, 
        # and hence reset() will be called
        return observation, reward, done, info
    def reset(self):
        self.env_id = 0
        obs = self.env_list[self.env_id].reset()
        return obs
    def render(self, mode='human'):
        return self.env_list[self.env_id].reset()
    def close (self):
        for env_id in range(self.n_envs):
            self.env_list[env_id].close()
