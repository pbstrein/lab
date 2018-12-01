import gym
import numpy as np
import torch
from torch import nn

from rl import *

import deepmind_lab


class DMLabEnvironmentFactory(EnvironmentFactory):
    def __init__(self):
        super(DMLabEnvironmentFactory, self).__init__()

    def new(self):
        return DMLabEnvironment()


class DMLabEnvironment(RLEnvironment):
    def __init__(self):
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
	   'fps': str(60),
           'width': str(80),
           'height': str(80),
	}
	self._env = deepmind_lab.Lab('tests/empty_room_test', ['RGB'], config=config)

    def step(self, action):
        """action is type np.ndarray of shape [1] and type np.uint8.
        Returns observation (np.ndarray), r (float), t (boolean)
        """
        #s, r, t, _ = self._env.step(action.item(), num_steps=self._num_steps)
        state = self.env.observations()
        reward = self.env.step(action.item(), num_steps=self._num_steps)
        terminated = not self._env.is_running()
        return state, reward, terminated

    def reset(self):
        """Returns observation (np.ndarray)"""
        self._env.reset()
        state = self._env.observations()
        return state['RGB']


class DMLabPolicyNetwork(nn.Module):
    """Policy Network for Deepmind Lab"""

    def __init__(self, state_dim=4, action_dim=2):
        super(DMLabPolicyNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, 10),
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
        scores = self._net(x)
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

    def __init__(self, state_dim=4):
        super(DMLabValueNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        """Receives an observation of shape [batch, state_dim].
        Returns the value of each state, in shape [batch, 1]
        """
        return self._net(x)


def main():
    factory = DMLabEnvironmentFactory()
    policy = DMLabPolicyNetwork()
    value = DMLabValueNetwork()
    ppo(factory, policy, value, multinomial_likelihood, epochs=1000, rollouts_per_epoch=100, max_episode_length=200,
        gamma=0.99, policy_epochs=5, batch_size=256)


if __name__ == '__main__':
    main()
