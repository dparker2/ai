from abc import ABC, abstractmethod
from collections import defaultdict
from random import choice, random
from pprint import pformat

import numpy as np


class Agent(ABC):
    """ Base Agent """

    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def explore(self, observation):
        pass

    @abstractmethod
    def act(self, observation):
        pass

    @abstractmethod
    def learn(self, prev_observation, action, next_observation, reward):
        pass


class Random(Agent):
    """ Random Agent """

    def explore(self, *args):
        return self.action_space.sample()

    def act(self, *args):
        return self.action_space.sample()

    def learn(self, *args):
        pass


class QLearning(Agent):
    """ QLearning Agent """

    learning_rate = 0.1
    discount_factor = 0.9

    def __init__(self, action_space):
        super().__init__(action_space)
        self.qtable = defaultdict(self._default_action_qs)

    def _default_action_qs(self):
        return np.array([0.0 for _ in range(self.action_space.n)])

    def _best_action(self, observation):
        """ Returns the best action and its q value for this observation"""
        action_qs = self.qtable[observation]

        action = choice(
            np.where(np.isclose(action_qs, action_qs.max()))[0]
        )  # Randomly break ties

        return action, action_qs[action]

    def explore(self, observation):
        """
        Act randomly to explore
        """
        return self.action_space.sample()

    def act(self, observation):
        """
        Best action given this observation?
        """
        action, _ = self._best_action(observation)
        return action

    def learn(self, prev_observation, action, next_observation, reward):
        """
        prev_observation: s
        action: a
        next_observation: s(t+1)
        reward: r

        Q(s,a) <-- Q(s,a) + α * (r + γ * maxQ(s(t+1),a) - Q(s,a))
        """
        old_q = self.qtable[prev_observation][action]
        _, optimal_next_q = self._best_action(next_observation)
        temporal_difference = (
            QLearning.discount_factor * optimal_next_q + reward - old_q
        )

        self.qtable[prev_observation][action] = (
            old_q + QLearning.learning_rate * temporal_difference
        )

    def __str__(self):
        s = "QLearning {\n"
        for k, v in self.qtable.items():
            s += f"\t{k}:\t{v}\n"
        s += "}"
        return s
