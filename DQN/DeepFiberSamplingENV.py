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

from helper_functions import reconnect_graph_generalized_version

class PolytopeENV(Env):
    def __init__(self, 
                 initial_states,
                 total_episodes, 
                 show_path_num, 
                 visited_states, 
                 basis_moves, 
                 node_num, 
                 P, 
                 lb):
        """
        Custom environment for Polytope problem, based on a multi-discrete action space.
        """
        super(PolytopeENV, self).__init__()

        # Environment settings
        self.initial_states = initial_states
        self.basis_move = basis_moves
        self.node_num = node_num
       
        self.P = P
        self.lb = lb
        self.visited_states = visited_states
        self.show_path_num = show_path_num
        self.total_episodes = total_episodes
        self.episode = -1
       
        # Action space (assuming each action component can take multiple discrete values)
        num_action_components = len(basis_moves)  # Number of dimensions in the action space
        self.action_space = spaces.MultiDiscrete([len(move) for move in self.basis_move])  # Define the multi-discrete action space

        # Observation space (assuming the state is represented by an array of integers)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=self.initial_states[0].shape, dtype=np.int32)

        self.reward_feasibility_list = []
        self.reward_non_zero_action_list = []
        self.reward_norm_list = []
        
        
        self.reset()

    def reset(self):
        """Reset the environment to its initial state."""
        self._iteration = 0
        self._total_reward = 0
        self.path = 0
        self.episode += 1
        
        
        # Start from a random visited state
        state_indx = 0
        
        state_indx = 0
        if len(self.initial_states.keys()) > 1:
            state_indx = random.randint(0,len(self.initial_states.keys())-1)
        state = self.initial_states[state_indx]
        self.state = state
 
        return self.state  # Return the initial state self.initial_states#

    def step(self, action):
        """Take a step in the environment."""
        action_rounded = np.array(np.round(action), dtype=int)
        action_rounded += self.lb
        all_actions = [np.multiply(action_rounded[i], self.basis_move[i]) for i in range(len(action_rounded))]
        all_actions = np.stack(all_actions)
        self.action = np.sum(all_actions, 0)

        self._iteration += 1
        done = False
        found_solution = False
        info = {}

        # Update the state
        next_state = np.add(self.state, self.action)

        # Compute reward components
        reward_feasibility = 0
        reward_non_zero_action = 0
       
        
        if np.all(self.action == 0):
            print("Action is a zero vector! Action coeffs: ", action_rounded)
            reward_non_zero_action = -10
            next_state = self.state
        else:
            reward_non_zero_action = 0

            if all(coord >= 0 for coord in next_state):
                
                state_g = self.create_state_graph(self.node_num, next_state)
                if next_state.tolist() not in self.visited_states.tolist():  
                    print("New state found!")
                    self.visited_states = np.concatenate((self.visited_states,[next_state]),axis=0)
                    reward_feasibility = 2
#                 else:
#                     reward_feasibility = -100
            else:
                for coord in next_state:
                    if coord < 0:
#                         reward_feasibility += 100 * coord
                        reward_feasibility = -100
                next_state = self.state

        reward = reward_feasibility + reward_non_zero_action
        self._total_reward += reward

#         self.reward_feasibility_list.append(reward_feasibility)
#         self.reward_non_zero_action_list.append(reward_non_zero_action)
#         self.reward_norm_list.append(reward_norm)
        
        # Define a done condition (e.g., maximum iterations)
        if self._iteration >= 100:  # You can define a suitable condition based on your problem
            print(f'Episode: {self.episode} ||| Reward: {self._total_reward} ||| Discovered States: {len(self.visited_states)}')
            done = True

        self.state = next_state

        return self.state, reward, done, info
    
    def _handle_close(self, evt):
        self._closed_plot = True

    
    # Given a state graph, construct a corresponding vector.
    def create_state_vector(self, g):
        adjacency = nx.to_numpy_array(g, sorted(list(g.nodes())), dtype=int)
        n_rows, n_cols = adjacency.shape
        upper_mask = np.triu(np.ones((n_rows, n_cols), dtype=bool), k=1)
        upper_diagonal = adjacency[upper_mask]
        flattened_vector = upper_diagonal.flatten()
        init_sol = np.array(flattened_vector)
        return init_sol
    
    # Given a state vector, construct a corresponding graph.
    def create_state_graph(self, num_nodes, state):
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        index = 0
        for n in range(num_nodes-1,-1,-1):
            for i in range(n+1):
                if num_nodes-1-n+i > num_nodes-1-n:
                    if state[index] != 0:
                        adj_matrix[num_nodes-1-n, num_nodes-1-n+i] = state[index]
                    #print(num_nodes-1-n, num_nodes-1-n+i)
                    index += 1
                    
        adj_matrix += np.triu(adj_matrix, 1).T
        MG = nx.MultiGraph()
        for i in range(num_nodes):
            for j in range(i, num_nodes):  # Iterate over upper triangular part including diagonal
                for _ in range(adj_matrix[i, j]):
                    MG.add_edge(i, j)
        return MG 
    
    # Check whether the state grah has the appropriate number of connected components. 
    def check_connectivity(self, g, p_num):
        conn_components = list(nx.connected_components(g))
        if len(conn_components) == p_num:
            return True
        else:
            return False
        
        
    def update_best_states(self, best_states, dict_size, state, state_directional_rew):
        
        keys = list(best_states.keys())
        states = [s[0].tolist() for s in best_states.values()]
        if state.tolist() in states: # do not reapet the same states. 
            return best_states
        #find the worst state in the dicitonary.
        key_with_min_float = min(best_states, key=lambda k: best_states[k][1])
        min_value = best_states[key_with_min_float][1]
        # Check for uniqness of states.
        if len(keys) >= dict_size:
            if min_value < state_directional_rew:  
                print("The BEST STATES DICT is full, and we found a better state, remove the worst state ", len(keys))
                best_states.pop(key_with_min_float)
                best_states[key_with_min_float] = (state, state_directional_rew)
        else:
            print("The BEST STATES DICT is not full, add state at key ", max(keys)+1)
            best_states[max(keys)+1] = (state, state_directional_rew)
        return best_states