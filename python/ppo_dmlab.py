
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn

import argparse

from rl import *

import deepmind_lab
import six
import os

import networks as nets


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

    obv_type = 'RGB_INTERLEAVED'

    ACTIONS = {
      'look_left': _action(-40, 0, 0, 0, 0, 0, 0),
      'look_right': _action(40, 0, 0, 0, 0, 0, 0),
      'look_up': _action(0, 20, 0, 0, 0, 0, 0),
      'look_down': _action(0, -20, 0, 0, 0, 0, 0),
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
	   'fps': str(fps),
           'width': str(height),
           'height': str(width),
	}

        #room = 'tests/empty_room_test'
        #room = 'demos/set_instruction'
        room = 'seekavoid_arena_01'
	#self._env = deepmind_lab.Lab(room, ['RGB_INTERLEAVED'], config=config)
	#self._env = deepmind_lab.Lab(room, ['RGB'], config=config)
	self._env = deepmind_lab.Lab(room, [self.obv_type], config=config)

        #self._action_spec = self._env.action_spec()
        #self.indices = {a['name']: i for i, a in enumerate(self._action_spec)}
        #self.mins = np.array([a['min'] for a in self._action_spec])
        #self.maxs = np.array([a['max'] for a in self._action_spec])

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
        reward *= 10
        terminated = not self._env.is_running()
        return state[self.obv_type], reward, terminated
        #return state['RGB'], reward, terminated
        #return state['RGB_INTERLEAVED'], reward, terminated

    def reset(self):
        """Returns observation (np.ndarray)"""
        self._env.reset()
        state = self._env.observations()
        #state = state['RGB']
        state = state[self.obv_type]
        print("state:" , state.shape)
        #state = state['RGB_INTERLEAVED']
        return state

    def is_running(self):
        return self._env.is_running()

    def get_observation(self):
        return self._env.observations()[self.obv_type]
        #return self._env.observations()['RGB']

    def get_screen_size(self):
        self._env.reset()
        observation_list = self._env.observation_spec()
        for val in observation_list:
            #if val['name'] == 'RGB':
            if val['name'] == self.obv_type:
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

    #def __init__(self, screen_height, screen_width, state_dim=19200, action_dim=4):
    def __init__(self, input_dim, state_dim=19200, action_dim=4):
        super(DMLabPolicyNetwork, self).__init__()
        self._conv_net_out_channels = 256 
	self._input_dim = input_dim

        # convolutional network to read in the pixels
        self._conv_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(64, self._conv_net_out_channels, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
        )
        self._net = nn.Sequential(
            #nn.Linear(self._conv_net_out_channels*screen_height*screen_width, 10),
	    nn.Linear(self._input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, action_dim)
        )
        '''
        self._net = nn.Sequential(
            nn.Linear(self._conv_net_out_channels*screen_height*screen_width, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, action_dim)
        )
        '''
        self._softmax = nn.Softmax(dim=1)
        

    def forward(self, x, get_action=True, explore=False):
        """Receives input x of shape [batch, state_dim].
        Outputs action distribution (categorical distribution) of shape [batch, action_dim],
        as well as a sampled action (optional).
        """
        #print("x in policy net: ", x.size())
        #conv_value = self._conv_net(x) # run convolutions on the pixels
        #b, c, h, w = conv_value.size()
        #conv_value = conv_value.view(b, c*h*w) # flatten to be fed into the linear network
        #scores = self._net(conv_value)

	scores = self._net(x)

        if explore:
            scores /= 10 # shrink by a factor of 10 so the scores are closer together, forces the network to choose more randomly

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

    #def __init__(self, screen_height, screen_width, state_dim=19200):
    def __init__(self, input_dim, state_dim=19200):
        super(DMLabValueNetwork, self).__init__()
        self._conv_net_out_channels = 64
	self._input_dim = input_dim

        # convolutional network to read in the pixels
        self._conv_net = nn.Sequential(
                 
            nn.Conv2d(3, 16, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32, self._conv_net_out_channels, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
        )

        '''
        # linear network to evaluate the value
        self._net = nn.Sequential(
            nn.Linear(self._conv_net_out_channels*screen_height*screen_width, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        '''
        self._net = nn.Sequential(
            #nn.Linear(self._conv_net_out_channels*screen_height*screen_width, 10),
	    nn.Linear(self._input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
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
        #conv_value = self._conv_net(x) # run convolutions on the pixels
        #b, c, h, w = conv_value.size()
        #conv_value = conv_value.view(b, c*h*w) # flatten to be fed into the linear network
        #value = self._net(conv_value)
	value = self._net(x)
        #return self._net(x)
        return value


def main(length, width, height, fps, level, train, save_model_loc, load_model_loc):
    print("length: ", length)
    print("width: ", width)
    print("height: ", height)
    print("fps: ", fps)
    print("level: ", level)
    print("train: ", train)
    print("save_model_loc: ", save_model_loc)
    print("load_model_loc: ", load_model_loc)

    factory = DMLabEnvironmentFactory(fps=fps, height=height, width=width)
    game_instance = factory.new()
    screen_size = game_instance.get_screen_size()
    actions = game_instance.get_actions()
    print("actions: ", actions)
    num_actions = game_instance.get_num_actions()
    print("num_actions: ", num_actions)
    game_instance.reset()

    conv = nets.ConvNetwork128(height, width, 3)

    #policy = DMLabPolicyNetwork(height, width, state_dim=screen_size, action_dim=num_actions)
    #value = DMLabValueNetwork(height, width, state_dim=screen_size)

    policy = DMLabPolicyNetwork(conv.get_output_dim(), action_dim=num_actions)
    value = DMLabValueNetwork(conv.get_output_dim())

    conv_file_name = 'conv-network'
    policy_file_name = 'policy-network'
    value_file_name = 'value-network'

    if load_model_loc:
        CONV_LOC = load_model_loc + conv_file_name
        POLICY_LOC = load_model_loc + policy_file_name
        VALUE_LOC = load_model_loc + value_file_name

        print("loading conv network from : ", CONV_LOC)
        conv.load_state_dict(torch.load(CONV_LOC))

        print("loading policy network from : ", POLICY_LOC)
        policy.load_state_dict(torch.load(POLICY_LOC))

        print("loading value network from: ", VALUE_LOC)
        value.load_state_dict(torch.load(VALUE_LOC))


    if train:
        '''
        factory = DMLabEnvironmentFactory(fps=fps, height=height, width=width)
        game_instance = factory.new()
        screen_size = game_instance.get_screen_size()
        actions = game_instance.get_actions()
        print("actions: ", actions)
        num_actions = game_instance.get_num_actions()
        print("num_actions: ", num_actions)
        game_instance.reset()
        policy = DMLabPolicyNetwork(height, width, state_dim=screen_size, action_dim=num_actions)
        value = DMLabValueNetwork(height, width, state_dim=screen_size)
        '''
        csv_loc = save_model_loc + "train-details.csv"
        #ppo(factory, policy, value, multinomial_likelihood, epochs=5, rollouts_per_epoch=5, max_episode_length=length,
            #gamma=0.99, policy_epochs=3, batch_size=256, lr=1e-4, weight_decay=0.0, environment_threads=2)
        #ppo(factory, policy, value, multinomial_likelihood, epochs=2, rollouts_per_epoch=1, max_episode_length=length,
            #gamma=0.99, policy_epochs=2, batch_size=256, lr=1e-4, weight_decay=0.0, environment_threads=2, data_loader_threads=2)
        ppo(factory, policy, value, multinomial_likelihood, embedding_net=conv, epochs=10, rollouts_per_epoch=7, max_episode_length=length,
            gamma=0.99, policy_epochs=5, batch_size=256, lr=1e-3, weight_decay=0.0, environment_threads=2, data_loader_threads=2, save_model=save_model_loc,
            csv_file=csv_loc)
        #ppo(factory, policy, value, multinomial_likelihood, embedding_net=conv, epochs=1, rollouts_per_epoch=1, max_episode_length=length,
            #gamma=0.99, policy_epochs=3, batch_size=256, lr=1e-3, weight_decay=0.0, environment_threads=2, data_loader_threads=2, save_model=save_model_loc,
            #csv_file=csv_loc)

        if save_model_loc:
            CONV_SAVE_LOC = save_model_loc + conv_file_name
            POLICY_SAVE_LOC = save_model_loc + policy_file_name
            VALUE_SAVE_LOC = save_model_loc + value_file_name

            print("saving conv network to: ", CONV_SAVE_LOC)
            torch.save(conv.state_dict(), CONV_SAVE_LOC)

            print("saving policy network to: ", POLICY_SAVE_LOC)
            torch.save(policy.state_dict(), POLICY_SAVE_LOC)

            print("saving value network to: ", VALUE_SAVE_LOC)
            torch.save(value.state_dict(), VALUE_SAVE_LOC)


    else:
	config = {
	   'fps': str(fps),
           'width': str(height),
           'height': str(width),
	}
        reward = 0
        for _ in six.moves.range(length):
            if not game_instance.is_running():
                print('Environment stopped early')
                env.reset()
                #agent.reset()
            obs = game_instance.get_observation()
	    obs = torch.from_numpy(obs).float().unsqueeze(0)
	    result = conv(obs)
            #result = torch.from_numpy(result).float().unsqueeze(0) # add batch dimension so it can go into the network
            #result = torch.from_numpy(obs).float().unsqueeze(0) # add batch dimension so it can go into the network
            probs, actions = policy(result)
            print("probs: ", probs)
            print("actions: ", actions)
            next_state, new_reward, terminated = game_instance.step(actions)
            reward += new_reward

        print('Finished after %i steps. Total reward received is %f'
            % (length, reward))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=bool, default=False,
                      help='Tell the agent to train or just run')
    parser.add_argument('--save-model', type=str, default=None,
                      help='Directory where the pytorch model will be saved')
    parser.add_argument('--load-model', type=str, default=None,
                      help='Directory where the pytorch model will be loaded from')
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
    main(args.length, args.width, args.height, args.fps, args.level_script,
            args.train, args.save_model, args.load_model)
