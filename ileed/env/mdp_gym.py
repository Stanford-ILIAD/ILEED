import numpy as np
import gym

import matplotlib.pyplot as plt
from matplotlib import colors

from functools import partial

# These three classes can be grouped under one 
# Entity class. 
class Agent():
    def __init__(self, id):
        super(Agent, self).__init__()
        self.id = id
        self.loc = None

class Apple():
    def __init__(self, id):
        super(Apple, self).__init__()
        self.id = id
        self.loc = None
        
class Bomb():
    def __init__(self, id):
        super(Bomb, self).__init__()
        self.id = id
        self.loc = None

class mdp_w_exp(gym.Env):

    def __init__(self, T, T_p, seed, dim=5, v_dim=5, R=1, n_goals=[10,50], obs_method='centergrid', start_phase=1):
        '''
        Custom single-agent MDP environment with two phases:

        (1) Green Phase (phase = 1)
            Some of the tiles will be green (apples). Your job is to 
            eat as many apples as you can. If you step on an apple, 
            you will receive a positive reward, and the next turn
            the apple will respawn elsewhere. 

        (2) Red Phase (phase = -1)
            Some of the tiles will be red (bombs). Your job is to 
            stay away from the bombs as every turn they will move in a
            random cardinal direction. If you step on a bomb, you will
            recieve a negative reward, and the bomb will respawn elsewhere. 

        There are 4 possible Actions (4 cardinal directions)
            0 - right
            1 - up
            2 - left
            3 - down

        Inputs:
            T           - Total steps in one episode
            T_p         - # of steps between phase transitions
            seed        - seed for numpy rng
            dim         - grid dimension
            v_dim       - dimension of vision grid
            R           - Reward value for both phases when goal is pressed
            n_goals     - number of goals per phase (2 dimensional list)
            obs_method  - one of 'lin', 'grid', 'centergrid'
            start_phase - deafult to '1' for apples, '-1' for bombs



        This works similairly to a gym.env(), same functions. 
        '''
        # Define INPUTS
        super(mdp_w_exp, self).__init__()
        self.T = T
        self.T_p = T_p
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.R = R
        self.n_goals = n_goals
        self.start_phase = start_phase

        assert(v_dim % 2 == 1 and v_dim <= 2*self.dim - 1) # use an odd number vision size
        self.v_dim = v_dim

        # we only have one agent, and 4 bombs and apples
        self.agent = Agent(id=0)
        self.apples = [Apple(i) for i in range(self.n_goals[0])]
        self.bombs = [Bomb(i) for i in range(self.n_goals[1])]

        # map from action to direction
        self.move_map = np.array([[0,1],[-1,0],[0,-1],[1,0]])
        # start in phase 1 (green apples) by default
        self.phase = start_phase 
        self.t = 1
        self.act_dim = 4

        # here we build the observation method, using a pointer
        self.obs_method = obs_method
        self.set_obs_method() # reset() is called here

        # setup rendering
        self.cmap, self.bounds, self.norm = self.get_colors()

        # gym
        self.action_space = gym.spaces.Discrete(self.act_dim)
        self.observation_space = gym.spaces.Box(low=-1, high=2, shape=self.obs_dim, dtype=np.float32)
    #---------------------------------------------------------------------#

    def reset(self):
        '''
        This reset is responsible for the whole environment
        '''
        self.t = 1
        self.phase = self.start_phase 
        # only hard reset moves the agent
        self.agent.loc = np.array(self.rng.choice(self.dim,2),dtype=np.int8)
        self.init_phase()
        return self.get_obs()

    def init_phase(self):
        '''
        Re-initializes based on the phase we are in.
        Store all the internal locations as np.int. 
        '''
        # first remove agents location
        temp = np.delete(np.arange(self.dim**2), 
                np.arange(self.dim**2)==(self.agent.loc[0]*self.dim+self.agent.loc[1]))
        # now choose the other locations 
        if self.phase == 1: # apples
            for i in range(self.n_goals[0]):
                tiles = self.rng.choice(temp, self.n_goals[0], replace=False)
                self.apples[i].loc = np.array([tiles[i]//self.dim,tiles[i]%self.dim],dtype=np.int8)
        else: # bombs
            for i in range(self.n_goals[1]):
                tiles = self.rng.choice(temp, self.n_goals[1], replace=False)
                self.bombs[i].loc = np.array([tiles[i]//self.dim,tiles[i]%self.dim],dtype=np.int8)

    def move_entity(self, entity, act):
        '''
        Given the act of the entity (agent or goal), 
        moves them to their new locs.

        Note that we clip the new location based on grid boundaries
        '''
        # potential new location
        new_loc = entity.loc + self.move_map[act] 
        entity.loc = np.array(np.clip(new_loc,np.zeros(2),np.ones(2)*(self.dim-1)),dtype=np.int8)

    def reset_entity(self, entity):
        '''
        resets the position of an individual entity, 
        making sure it is not in the same tile as any other entity
        '''
        locs = np.arange(self.dim**2)
        # remove locs from the set
        locs = np.delete(locs,locs==(self.dim*self.agent.loc[0]+self.agent.loc[1])) 
        if self.phase == 1: # apples
            for apple in self.apples:
                locs = np.delete(locs,locs==(self.dim*apple.loc[0]+apple.loc[1]))

        else: # bombs
            for bomb in self.bombs:
                locs = np.delete(locs,locs==(self.dim*bomb.loc[0]+bomb.loc[1]))

        # choose from the remaining set
        loc = self.rng.choice(locs)
        entity.loc = np.array([loc//self.dim,loc%self.dim],dtype=int)
        return

    def move_goals(self):
        '''
        Every turn the apples/bombs move randomly.
        NOTE: As of now only the bombs move, and they are 
        allowed to occupy the same tile
        '''
        if self.phase == 1:
            pass
            # for goal in self.apples:
            #     self.move_entity(goal, self.rng.choice(self.act_dim))
        else:
            for goal in self.bombs:
                self.move_entity(goal, self.rng.choice(self.act_dim))
        return
    
    def check_goal(self):
        '''
        Checks if the agent has stepped on an apple/bomb.

        IF so, resets the location of that goal
        AND returns reward (either way)
        '''
        if self.phase == 1:
            # default reward = 0
            rew = 0
            for goal in self.apples:
                # agent is on apple
                if np.all(self.agent.loc == goal.loc):
                    rew = self.R
                    # reset its position
                    self.reset_entity(goal)
                    break
        else:
            # default reward is positive here
            rew = 1
            for goal in self.bombs:
                # agent is on a bomb
                if np.all(self.agent.loc == goal.loc):
                    rew = -1*self.R
                    # reset its position
                    self.reset_entity(goal)
                    break
        return rew

    def step(self, act):
        '''
        This returns (obs, rew, prev_act, done) tuple:
            obs      - 3-Channel, 2-dimensional
                            obs[0]:     Grid Layout 0-valid location, 1-player location   
                            obs[2,3]:   apple/bomb locations. For bombs, the integer
                                        represents how many bombs are in that location. 

            rew      - float representing reward at step t
            done     - Bool that is flagged True when T steps have passed (not T_r)
            info     - empty {} for now 
        '''
        # move the agent and the goals
        self.move_entity(self.agent ,act)
        self.move_goals()
        rew = self.check_goal()

        # update time and flip to next phase if needed
        self.t += 1
        if self.t%self.T_p == 1:
            self.phase *= -1
            self.init_phase()

        return (self.get_obs(), rew, self.t>self.T, {})
    
    def get_colors(self):
        '''
        white   - 0 empty space
        purple  - 1 agent
        green   - 2 apple
        red     - 3 bomb
        '''

        cmap = colors.ListedColormap(['white','purple','green','red'])
        # We want to correspond each of these colors 
        # to a value of 0,1,2,3,4,5 respecitvely
        bounds = [-0.5,0.5,1.5,2.5,3.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        return(cmap, bounds, norm)

    def get_render_data(self):
        '''
        Returns grid representing the environment. 
        
        The integer values correspond to the entities 
        defines in self.get_colors()
        '''
        # Initialize the grid to be empty, with no agents or landmarks
        
        # goals
        grid = np.zeros((self.dim,self.dim),dtype=np.int8)
        # agent loc
        grid[self.agent.loc[0],self.agent.loc[1]] = 1
        # apples
        if self.phase == 1:
            for apple in self.apples:
                grid[apple.loc[0],apple.loc[1]] = 2 
        # bombs
        else:
            for bomb in self.bombs:
                grid[bomb.loc[0],bomb.loc[1]] = 3
                
        return grid

    def set_obs_method(self):
        # contents of reset() are placed here as well 
        self.agent.loc = np.array(self.rng.choice(self.dim,2),dtype=np.int8)
        self.init_phase()
        # now we condition which function we point to
        if self.obs_method == 'grid':
            self.obs_pointer = self.get_obs_grid
            temp = self.reset()
            self.obs_dim = np.array([3,self.dim,self.dim])
        elif self.obs_method == 'centergrid':
            # self.obs_pointer = self.get_obs_grid
            self.obs_pointer = partial(self.get_obs_grid_centered_at_agent)
            self.reset()
            self.obs_dim = np.array([2,self.v_dim,self.v_dim])

    def get_obs(self):
        return self.obs_pointer()

    def get_obs_grid(self):
        '''
        obs gets built on demand from the locs of objects in env. 
        Returns a 3-channel dim x dim grid: (agent_loc, apples, bombs)
        '''
        layout = np.zeros((3,self.dim,self.dim),dtype=np.uint8)
        # agent loc
        layout[0,self.agent.loc[0],self.agent.loc[1]] = 1
        # apples
        if self.phase == 1:
            for apple in self.apples:
                layout[1,apple.loc[0],apple.loc[1]] = 1 
        # bombs
        else:
            for bomb in self.bombs:
                layout[2,bomb.loc[0],bomb.loc[1]] += 1

        return layout

    def get_obs_grid_centered_at_agent(self):
        '''
        obs gets built on demand from the locs of objects in env.
        Returns a 2-channel dim x dim grid: (agent_loc, apples, bombs)

        wrapper-grid centers the grid at the agent location, by building a (2*dim-1) X (2*dim-1) grid
        then we trim off the edges of this wrapper-grid to return a centered grid with size=v_dim
        e.g. if v_dim=5 we return a 5x5 grid with the agent in the middle cell
        '''

        wrapper = -1*np.ones((2,2*self.dim-1,2*self.dim-1),dtype=float)

        TL_corner = (self.dim-1 - self.agent.loc[0], self.dim-1 - self.agent.loc[1])
        BR_corner = TL_corner[0]+self.dim, TL_corner[1]+self.dim
        wrapper[:,TL_corner[0]:BR_corner[0], TL_corner[1]:BR_corner[1]] = self.get_obs_grid()[1:3]

        TL_corner = (2*self.dim-1 - self.v_dim) // 2
        layout = wrapper[:,TL_corner:TL_corner+self.v_dim, TL_corner:TL_corner+self.v_dim]

        return layout
##---------------------------------------------------------------------------------
    # finally the render function, it is a little sloppy as every iteration 
    # I delete the figure after displaying it on the screen for a little bit. Still
    # Matplotlib is running in interactive mode, allowing for excecution to continue 
    # even after plt.show(), the only reason for the pause is so that we can actually
    # see what is happening and not jump to next step right away. 
##---------------------------------------------------------------------------------
    def render(self, mode='human'):
        # setup the figure
        fig = plt.figure(figsize=(3,3))
        ax = fig.subplots()
        # draw gridlines 
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.margins(0)
        fig.tight_layout(pad=0)
        ax.set_xticks(np.arange(-.5, self.dim, 1));
        ax.set_yticks(np.arange(-.5, self.dim, 1));
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        # retrieve the appropriate grid data
        grid = self.get_render_data()
        ax.imshow(grid, cmap = self.cmap, norm = self.norm)

        # depending on mode, render it or return RGB image
        if mode == 'human':
            plt.show()
            plt.pause(5)
            plt.close('all')
        elif mode == 'rgb_array':
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close('all')
            return image_from_plot

    def render_ascii(self):
        grid = self.get_obs()
        print(grid)
