import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
from operator import add
from gym import spaces, Env
import os
import math as m
import networkx as nx
import pickle
import sys

from helper_functions import create_state_vector

from gerrychain import (Partition, Graph,
                        updaters, constraints, accept, metagraph)
from gerrychain.proposals import recom
from gerrychain.constraints import contiguous
from functools import partial



class PolytopeENV(Env):
    def __init__(self, 
                 initial_state,
                 initial_partition,
                 show_path_num,
                 partition_num,
                 node_num):
        super(PolytopeENV, self).__init__()

        
        # Environment settings
        self.initial_state = initial_state
        self.initial_partition = initial_partition
        self.best_partition = initial_partition
        self.best_reward = self.calculate_reward(self.best_partition)
        self.show_path_num = show_path_num
        self.partition_num = partition_num
        self.node_num = node_num
        self.episode = -1
        self.reset()

    def reset(self):
        """Reset the environment to its initial state."""
        self._iteration = 0
        self._total_reward = 0
        self.path = 0
        self.episode += 1
        
        state = self.initial_state
        self.state = state
        self.state_partition = self.initial_partition

        return self.state, self.state_partition  # Return the initial state self.initial_states#



    def step(self, action):
        """Take a step in the environment."""

        self._iteration += 1
        done = False
        info = {}

        # Update the state
        self.next_state_partition = self.apply_action(self.state_partition, action)
        
        reward_diff = self.calculate_reward_diff(self.state_partition, self.next_state_partition)
        reward = self.calculate_reward(self.state_partition)
        if reward > self.best_reward:
            self.best_partition = self.state_partition
            self.best_reward = reward
        self._total_reward += reward
        
        # Define a done condition (e.g., maximum iterations)
        if self._iteration % 1000 == 0:  # You can define a suitable condition based on your problem
            print(f'Step: {self._iteration} ||| Total reward: {self._total_reward}')

        self.state = create_state_vector(self.node_num, self.partition_num, self.next_state_partition.parts)
        self.state_partition = self.next_state_partition

        return self.state, self.state_partition, reward_diff, done, info
    

    def apply_action(self, state_partition, action):
        next_partition = Partition(flips=action, parent=state_partition)
        return next_partition
    
    def calculate_reward_diff(self, state_partition, next_state_partition):
        reward_dict = constraints.validity.deviation_from_ideal(partition=state_partition)            
        reward = sum([v for v in set(reward_dict.values())])

        new_reward_dict = constraints.validity.deviation_from_ideal(partition=next_state_partition)            
        new_reward = sum([v for v in set(new_reward_dict.values())])
        return new_reward - reward
    
    def calculate_reward(self, state_partition):
        reward_dict = constraints.validity.deviation_from_ideal(partition=state_partition)            
        reward = sum([v for v in set(reward_dict.values())])

        return reward
    
    def get_best_partition(self):
        return self.best_partition, self.best_reward


   