from abc import ABCMeta, abstractmethod
import random

import gym


class BaseAgent(metaclass=ABCMeta):

    @abstractmethod
    def sample_action(self, state):
        pass


class RandomAgent(BaseAgent):
    def __init__(self, action_space, num_actions=1, seed=None):
        self.action_space = action_space
        self.num_actions = num_actions
        if seed is not None:
            self.action_space.seed(seed)

    def sample_action(self, state=None):
        s = self.action_space.sample()
        action = {}
        selected_actions = random.sample(list(s), self.num_actions)
        for sample in selected_actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = s[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = s[sample]
                # if str(self.action_space[sample]) == 'Discrete(2)':
                #     action[sample] = bool(s[sample])
                # else:
                #     action[sample] = s[sample]
        return action

class AdhocAgent(BaseAgent):
    def __init__(self, action_space, num_actions=1):
        self.action_space = action_space
        self.num_actions = num_actions

    def sample_action(self, state=None):
        actions = []
        actions.append({'extend-to-shelf': 1.0})
        actions.append({'right-shift': 2.5})
        actions.append({'extend-arm': 1.75})
        actions.append({'pick-up___c0__s1': 1.0})
        actions.append({'retract-off-shelf': 1.0})
        actions.append({'change-shelf___s1__s2': 1.0})
        actions.append({'extend-to-shelf': 1.0})
        actions.append({'put-down___c0__s2': 1.0})
        
        return actions[state]

class NoOpAgent(BaseAgent):
    def __init__(self, action_space, num_actions=0):
        self.action_space = action_space
        self.num_actions = num_actions

    def sample_action(self, state=None):
        action = {}
        return action

