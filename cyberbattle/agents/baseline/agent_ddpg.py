# %%
import os
from numpy import ndarray
from cyberbattle._env import cyberbattle_env
import numpy as np
from typing import List, NamedTuple, Optional, Tuple, Union
import random

# deep learning packages
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torch.cuda

from learner import Learner
from agent_wrapper import EnvironmentBounds
import cyberbattle.agents.baseline.agent_wrapper as w
from agent_randomcredlookup import CredentialCacheExploiter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CyberBattleStateActionModel:
    """ Define an abstraction of the state and action space
        for a CyberBattle environment, to be used to train a Q-function.
    """

    def __init__(self, ep: EnvironmentBounds):
        self.ep = ep

        self.global_features = w.ConcatFeatures(ep, [
            # w.Feature_discovered_node_count(ep),
            # w.Feature_owned_node_count(ep),
            w.Feature_discovered_notowned_node_count(ep, None)

            # w.Feature_discovered_ports(ep),
            # w.Feature_discovered_ports_counts(ep),
            # w.Feature_discovered_ports_sliding(ep),
            # w.Feature_discovered_credential_count(ep),
            # w.Feature_discovered_nodeproperties_sliding(ep),
        ])

        self.node_specific_features = w.ConcatFeatures(ep, [
            # w.Feature_actions_tried_at_node(ep),
            w.Feature_success_actions_at_node(ep),
            w.Feature_failed_actions_at_node(ep),
            w.Feature_active_node_properties(ep),
            w.Feature_active_node_age(ep)
            # w.Feature_active_node_id(ep)
        ])

        self.state_space = w.ConcatFeatures(ep, self.global_features.feature_selection +
                                            self.node_specific_features.feature_selection)

        self.action_space = w.AbstractAction(ep)

    def get_state_astensor(self, state: w.StateAugmentation):
        state_vector = self.state_space.get(state, node=None)
        state_vector_float = np.array(state_vector, dtype=np.float32)
        state_tensor = torch.from_numpy(state_vector_float).unsqueeze(0)
        return state_tensor

    def implement_action(
            self,
            wrapped_env: w.AgentWrapper,
            actor_features: ndarray,
            abstract_action: np.int32) -> Tuple[str, Optional[cyberbattle_env.Action], Optional[int]]:
        """Specialize an abstract model action into a CyberBattle gym action.

            actor_features -- the desired features of the actor to use (source CyberBattle node)
            abstract_action -- the desired type of attack (connect, local, remote).

            Returns a gym environment implementing the desired attack at a node with the desired embedding.
        """

        observation = wrapped_env.state.observation

        # Pick source node at random (owned and with the desired feature encoding)
        potential_source_nodes = [
            from_node
            for from_node in w.owned_nodes(observation)
            if np.all(actor_features == self.node_specific_features.get(wrapped_env.state, from_node))
        ]

        if len(potential_source_nodes) > 0:
            source_node = np.random.choice(potential_source_nodes)

            gym_action = self.action_space.specialize_to_gymaction(
                source_node, observation, np.int32(abstract_action))

            if not gym_action:
                return "exploit[undefined]->explore", None, None

            elif wrapped_env.env.is_action_valid(gym_action, observation['action_mask']):
                return "exploit", gym_action, source_node
            else:
                return "exploit[invalid]->explore", None, None
        else:
            return "exploit[no_actor]->explore", None, None

# %%

# DDPG Algorithm
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1


    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal
