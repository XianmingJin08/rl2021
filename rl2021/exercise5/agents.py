from abc import ABC, abstractmethod
from collections import defaultdict
import random
import sys
from typing import List, Dict, DefaultDict

import numpy as np
from gym.spaces import Space, Box
from gym.spaces.utils import flatdim

from rl2021.exercise5.matrix_game import actions_to_onehot


def obs_to_tuple(obs):
    return tuple([tuple(o) for o in obs])


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    **DO NOT CHANGE THIS BASE CLASS**

    """

    def __init__(
            self,
            num_agents: int,
            action_spaces: List[Space],
            observation_spaces: List[Space],
            gamma: float,
            **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param observation_spaces (List[Space]): observation spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self, obs: List[np.ndarray]) -> List[int]:
        """Chooses an action for all agents given observations

        :param obs (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
    """

    def __init__(self, learning_rate: float = 0.5, epsilon: float = 1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents


        :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]

    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        actions = []
        ### PUT YOUR CODE HERE ###
        for i, obs in enumerate(obss):
            q_table = self.q_tables[i]
            n_acts = self.n_acts[i]
            act_vals = [q_table[obs, action] for action in range(n_acts)]
            max_val = max(act_vals)
            max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]
            ### RETURN AN ACTION HERE ###
            if random.random() < self.epsilon:
                actions.append(random.randint(0, n_acts - 1))
            else:
                actions.append(random.choice(max_acts))
        return actions

    def learn(
            self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray],
            dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the next environmental state for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###
        for i, obs in enumerate(obss):
            q_table = self.q_tables[i]
            action = actions[i]
            reward = rewards[i]
            n_obs = n_obss[i]
            done = dones[i]
            n_acts = self.n_acts[i]
            q_value_old = q_table[obs, action]
            next_best_q_value = max([q_table[n_obs, a] for a in range(n_acts)]) if not done else 0
            q_table[obs, action] = q_value_old + self.learning_rate * (
                    reward + self.gamma * next_best_q_value - q_value_old)
            self.q_tables[i] = q_table
            updated_values.append(q_table[(obs, action)])
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        max_deduct, decay = 0.95, 0.07
        self.epsilon = 1.0 - (min(1.0, timestep / (decay * max_timestep))) * max_deduct


class JointActionLearning(MultiAgent):
    """Agents using the Joint Action Learning algorithm with Opponent Modelling

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
    """

    def __init__(self, learning_rate: float = 0.5, epsilon: float = 1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of
            observations and joint actions to respective Q-values for all agents
        :attr models (List[DefaultDict[DefaultDict]]): each agent holding model of other agent
            mapping observation to other agent actions to count of other agent action

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount
        rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: defaultdict(lambda: 0)) for _ in range(self.num_agents)]

        # count observations - count for each agent
        self.c_obss = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        joint_action = []
        ### PUT YOUR CODE HERE ###
        for i, obs in enumerate(obss):
            q_table = self.q_tables[i]
            ev = 1
            if self.c_obss[i][obs] == 0:
                ev = 0
            if random.random() < self.epsilon or ev == 0:
                joint_action.append(random.randint(0, self.n_acts[i] - 1))
            else:
                j = (i + 1) % 2
                n_acts = self.n_acts[i]
                n_acts_opp = self.n_acts[j]
                evs = []
                for action in range(n_acts):
                    ev_state_action = 0
                    for action_opp in range(n_acts_opp):
                        ev_state_action += (self.models[i][obs][action_opp] / self.c_obss[i][obs]) * q_table[obs,(
                            action, action_opp)]
                    evs.append(ev_state_action)
                max_ev = max(evs)
                max_acts = [idx for idx, act_ev in enumerate(evs) if act_ev == max_ev]
                joint_action.append(random.choice(max_acts))

        return joint_action

    def learn(
            self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray],
            dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the next environmental state for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###
        for i in range(self.num_agents):
            obs = obss[i]
            reward = rewards[i]
            n_obs = n_obss[i]
            done = dones[i]
            q_table = self.q_tables[i]
            j = (i + 1) % 2
            action_opp = actions[j]
            self.c_obss[i][obs] += 1 if self.c_obss[i][obs] else 1
            self.models[i][obs][action_opp] += 1 if self.models[i][obs][action_opp] else 1
            q_value_old = q_table[obs,tuple(actions)]
            evs = []
            for action_next in range(self.n_acts[i]):
                ev_state_actionNext = 0
                for action_next_opp in range(self.n_acts[j]):
                    ev_state_actionNext += (self.models[i][n_obs][action_next_opp] / self.c_obss[i][n_obs]) * q_table[n_obs,(action_next,action_next_opp)]
                evs.append(ev_state_actionNext)
            next_best_ev = max(evs) if not done else 0
            q_table[obs, tuple(actions)] = q_value_old + self.learning_rate * (reward + self.gamma * next_best_ev - q_value_old)
            self.q_tables[i] = q_table
            updated_values.append(q_table[obs, tuple(actions)])
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        max_deduct, decay = 0.95, 0.07
        self.epsilon = 1.0 - (min(1.0, timestep / (decay * max_timestep))) * max_deduct
