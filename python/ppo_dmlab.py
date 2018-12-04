import gym
import numpy as np
import torch
from torch import nn

import argparse

from rl import *

import deepmind_lab
import six


class DMLabEnvironmentFactory(EnvironmentFactory):
    def __init__(self, fps=60, height=480, width=640):
        self.fps = fps
        self.height = height
        self.width = width
        super(DMLabEnvironmentFactory, self).__init__()

    def new(self):
        return DMLabEnvironment(fps=self.fps, height=self.height, width=self.height)


def _action(*entries):
    return np.array(entries, dtype=np.intc)

class DMLabEnvironment(RLEnvironment):

    ACTIONS = {
      'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
      'look_right': _action(20, 0, 0, 0, 0, 0, 0),
      'look_up': _action(0, 10, 0, 0, 0, 0, 0),
      'look_down': _action(0, -10, 0, 0, 0, 0, 0),
      'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
      'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
      'forward': _action(0, 0, 0, 1, 0, 0, 0),
      'backward': _action(0, 0, 0, -1, 0, 0, 0),
      'fire': _action(0, 0, 0, 0, 1, 0, 0),
      'jump': _action(0, 0, 0, 0, 0, 1, 0),
      'crouch': _action(0, 0, 0, 0, 0, 0, 1)
    } 

    ACTION_LIST = list(six.viewvalues(ACTIONS))

    _action_spec = None
    def __init__(self, fps=60, height=480, width=640):
        super(DMLabEnvironment, self).__init__()
        #self._env = gym.make('CartPole-v0')
        self._num_steps = 1
	config = {
	  #'fps': str(fps),
	  #'width': str(width),
	  #'height': str(height)
	   #'fps': str(60),
           #'width': str(640),
           #'height': str(480),
	   'fps': str(fps),
           'width': str(height),
           'height': str(width),
	}
        #room = 'tests/empty_room_test'
        room = 'demos/set_instruction'
	#self._env = deepmind_lab.Lab(room, ['RGB_INTERLEAVED'], config=config)
	self._env = deepmind_lab.Lab(room, ['RGB'], config=config)

        self._action_spec = self._env.action_spec()
        self.indices = {a['name']: i for i, a in enumerate(self._action_spec)}
        self.mins = np.array([a['min'] for a in self._action_spec])
        self.maxs = np.array([a['max'] for a in self._action_spec])

    def step(self, action):
        """action is type np.ndarray of shape [1] and type np.uint8.
        Returns observation (np.ndarray), r (float), t (boolean)
        """
        #s, r, t, _ = self._env.step(action.item(), num_steps=self._num_steps)
        state = self._env.observations()
        # turn the action index into an array of actions
	action_choice = self.ACTION_LIST[action.item()]
	

        #reward = self._env.step(action.item(), num_steps=self._num_steps)
        reward = self._env.step(action_choice, num_steps=self._num_steps)
        terminated = not self._env.is_running()
        return state['RGB'], reward, terminated
        #return state['RGB_INTERLEAVED'], reward, terminated

    def reset(self):
        """Returns observation (np.ndarray)"""
        self._env.reset()
        state = self._env.observations()
        state = state['RGB']
        #state = state['RGB_INTERLEAVED']
        return state

    def get_screen_size(self):
        self._env.reset()
        observation_list = self._env.observation_spec()
        for val in observation_list:
            if val['name'] == 'RGB':
                obv = val
                break
        result =  np.prod([x for x in val['shape']])
        return result


    def get_actions(self):
        if not self._action_spec:
            self._action_spec = self._env.action_spec()
        return self._action_spec

    def get_num_actions(self):
        return len(self.ACTIONS.keys())


class DMLabPolicyNetwork(nn.Module):
    """Policy Network for Deepmind Lab"""

    def __init__(self, state_dim=19200, action_dim=4):
        super(DMLabPolicyNetwork, self).__init__()
        self._conv_net_out_channels = 64

        # convolutional network to read in the pixels
        self._conv_net = nn.Sequential(
                 
            nn.Conv2d(3, 16, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32, self._conv_net_out_channels, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
        )

        # linear network to make the policy from the output of the convolution
        self._net = nn.Sequential(
            nn.Linear(self._conv_net_out_channels*80*80, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, action_dim)
        )
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x, get_action=True):
        """Receives input x of shape [batch, state_dim].
        Outputs action distribution (categorical distribution) of shape [batch, action_dim],
        as well as a sampled action (optional).
        """
        #print("x in policy net: ", x.size())
        conv_value = self._conv_net(x) # run convolutions on the pixels
        b, c, h, w = conv_value.size()
        conv_value = conv_value.view(b, c*h*w) # flatten to be fed into the linear network
        scores = self._net(conv_value)
        probs = self._softmax(scores)

        if not get_action:
            return probs
	
        batch_size = x.shape[0]
        actions = np.empty((batch_size, 1), dtype=np.uint8)
        probs_np = probs.cpu().detach().numpy()
        for i in range(batch_size):
            action_one_hot = np.random.multinomial(1, probs_np[i])
            action_idx = np.argmax(action_one_hot)
            actions[i, 0] = action_idx
        return probs, actions


class DMLabValueNetwork(nn.Module):
    """Approximates the value of a particular DeepMind Lab state."""

    def __init__(self, state_dim=19200):
        super(DMLabValueNetwork, self).__init__()
        self._conv_net_out_channels = 64

        # convolutional network to read in the pixels
        self._conv_net = nn.Sequential(
                 
            nn.Conv2d(3, 16, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32, self._conv_net_out_channels, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
        )

        # linear network to evaluate the value
        self._net = nn.Sequential(
            nn.Linear(self._conv_net_out_channels*80*80, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        '''
        self._net = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        '''

    def forward(self, x):
        """Receives an observation of shape [batch, state_dim].
        Returns the value of each state, in shape [batch, 1]
        """
        conv_value = self._conv_net(x) # run convolutions on the pixels
        b, c, h, w = conv_value.size()
        conv_value = conv_value.view(b, c*h*w) # flatten to be fed into the linear network
        value = self._net(conv_value)
        #return self._net(x)
        return value


def main(length, width, height, fps, level, train):
    print("length: ", length)
    print("width: ", width)
    print("height: ", height)
    print("fps: ", fps)
    print("level: ", level)
    print("train: ", train)

    channels = 3
    #screen_size = height * width * 3
    factory = DMLabEnvironmentFactory(fps=fps, height=height, width=width)
    game_instance = factory.new()
    screen_size = game_instance.get_screen_size()
    actions = game_instance.get_actions()
    print("actions: ", actions)
    num_actions = game_instance.get_num_actions()
    print("num_actions: ", num_actions)
    policy = DMLabPolicyNetwork(state_dim=screen_size, action_dim=num_actions)
    value = DMLabValueNetwork(state_dim=screen_size)
    if train:
        ppo(factory, policy, value, multinomial_likelihood, epochs=1, rollouts_per_epoch=1, max_episode_length=200,
            gamma=0.99, policy_epochs=5, batch_size=256)
    else:

        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=bool, default=False,
                      help='Tell the agent to train or just run')
    parser.add_argument('--length', type=int, default=1000,
                      help='Number of steps to run the agent')
    parser.add_argument('--width', type=int, default=80,
                      help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=80,
                      help='Vertical size of the observations')
    parser.add_argument('--fps', type=int, default=60,
                      help='Number of frames per second')
    parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
    parser.add_argument('--level_script', type=str,
                      default='demos/set_instruction',
                      help='The environment level script to load')
    #parser.add_argument('--record', type=str, default=None,
                      #help='Record the run to a demo file')
    #parser.add_argument('--demo', type=str, default=None,
                      #help='Play back a recorded demo file')
    #parser.add_argument('--demofiles', type=str, default=None,
                      #help='Directory for demo files')
    #parser.add_argument('--video', type=str, default=None,
                      #help='Record the demo run as a video')

    args = parser.parse_args()
    #if args.runfiles_path:
        #deepmind_lab.set_runfiles_path(args.runfiles_path)
    #run(args.length, args.width, args.height, args.fps, args.level_script,
      #args.record, args.demo, args.demofiles, args.video)
    main(args.length, args.width, args.height, args.fps, args.level_script,
            args.train)
