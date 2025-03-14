import math as m
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import itertools
from functools import reduce
import random
from collections import deque
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from helper_functions import group_keys_by_value, create_action_vector

from gerrychain import (Partition, Graph, updaters, constraints, accept, metagraph)
from gerrychain.proposals import recom
from gerrychain.constraints import contiguous
from functools import partial






class DQNMulitDiscrete(nn.Module):
    def __init__(self, state_size, action_space, learning_rate):
        super(DQNMulitDiscrete, self).__init__()
        self.state_size = state_size
        self.action_space = action_space  # A list representing action space sizes for each dimension
        self.learning_rate = learning_rate
        
        # Shared layers
        self.fc1 = nn.Linear(state_size,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        
        # Separate output layers for each dimension of the action vector
        self.output_layers = nn.ModuleList([nn.Linear(64, action_size) for action_size in action_space])
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Loss function (using Huber Loss for stability)
        self.loss_function = nn.SmoothL1Loss()  # Alternatively, you can use nn.MSELoss()
        
    def forward(self, state):
        # Pass the state through the shared layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        
        # Output Q-values for each action dimension
        q_values = [output_layer(x) for output_layer in self.output_layers]
        return q_values  # This is a list of Q-values for each action component (multi-discrete output)

    


    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)  # Store experiences in the buffer
        self.batch_size = batch_size  # Number of experiences to sample during training

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from the buffer."""
        batch = random.sample(self.buffer, self.batch_size)
        return batch

    def size(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

    def can_sample(self):
        """Check if there are enough experiences to sample a batch."""
        return self.size() >= self.batch_size
    
   
    

class RewardBuffer:
    def __init__(self, buffer_size):
        self.reward_buffer = deque(maxlen=buffer_size)  # Store experiences in the buffer
        
    
    def add(self, reward):
        """Add an experience to the buffer."""
        self.reward_buffer.append(reward)
    
    def get_reward_mean_std(self):

        # Compute the mean and standard deviation of the rewards
        reward_mean = np.mean(self.reward_buffer)
        reward_std = np.std(self.reward_buffer)

        return reward_mean, reward_std
    
    
# Helper function for epsilon-greedy action selection
def select_action(q_network, state, state_partition, epsilon, district_num):
    """
    Epsilon-greedy action selection.
    :param q_network: The DQN network
    :param state: Current state
    :param epsilon: Exploration rate
    :param action_space: List of action space sizes for each dimension
    :return: A tuple of actions for each dimension
    """

    if np.random.rand() < epsilon:
        # Pick a random flip.
        single_flips = metagraph.all_valid_flips(state_partition, constraints=[contiguous])
        single_flips_list = list(single_flips)
        single_flips_list = [dict(t) for t in {tuple(d.items()) for d in single_flips_list}]
        single_flips_list = sorted(single_flips_list, key=lambda x: (list(x.values())[0], list(x.keys())[0]))
        available_flips_per_district = group_keys_by_value(single_flips_list)

        excluded_blocks = [value for key, value in state_partition.parts.items() if len(value) <= 2]
        excluded_blocks_list = list(set().union(*excluded_blocks))
        filtered_available_flips_per_district = {key: list(set(value)-set(excluded_blocks_list)) for key, value in available_flips_per_district.items() if len(list(set(value)-set(excluded_blocks_list))) > 0} 

        # Pick a random district.
        district_action = random.choice(list(filtered_available_flips_per_district.keys()))

        action_block = np.random.randint(0, len(filtered_available_flips_per_district[str(district_action)]))
        
        action = {filtered_available_flips_per_district[str(district_action)][action_block]: str(district_action)}


    else:
        # Greedy action selection for each component
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
        q_values = q_network(state_tensor)  # Get Q-values for each component

        # Select the action with the highest Q-value for each component
        action = [torch.argmax(q_vals).item() for q_vals in q_values]

        action = {action[1]:str(action[0]+1)}
        # print("NETWORK Best action: \n", action)

    return action


def select_optimal_action(q_network, state):
    # Greedy action selection for each component
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
    q_values = q_network(state_tensor)  # Get Q-values for each component

    # Select the action with the highest Q-value for each component
    action = [torch.argmax(q_vals).item() for q_vals in q_values]
    return action



def train_q_network(q_network, target_q_network, optimizer, scheduler, replay_buffer, gamma, action_space, step):
    """
    Trains the Q-network by sampling from the experience replay buffer.
    """
    
    if replay_buffer.can_sample() == False:
        return  # If there aren't enough experiences in the buffer, skip the update
    
    # print("Train the networks")
    training_regime = False

    # Sample a batch from the replay buffer
    batch = replay_buffer.sample()
    random.shuffle(batch)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert batch to tensors
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Compute Q-values for the current state
    q_values = q_network(states)

    # Select the Q-value corresponding to the chosen action for each component
    current_q_values = [q_values[i].gather(1, actions[:, i].unsqueeze(1)).squeeze(1) for i in range(len(action_space))]

    # Compute target Q-values using the target network
    with torch.no_grad():
        next_q_values = target_q_network(next_states)
        target_q_values = []
        for i, next_q_vals in enumerate(next_q_values):
            # Max Q-value for each action component in the next state
            max_next_q_value = torch.max(next_q_vals, dim=1)[0]
            target_q_value = rewards + (1 - dones) * gamma * max_next_q_value
            target_q_values.append(target_q_value)

    # Compute loss for each action component
    losses = [q_network.loss_function(current_q_values[i], target_q_values[i]) for i in range(len(action_space))]
    
    # Combine the losses (sum) and backpropagate
    total_loss = sum(losses)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    #scheduler.step()
    

# Update target network (copy weights)
def update_target_network(q_network, target_q_network):
    target_q_network.load_state_dict(q_network.state_dict())  # Copy weights from primary to target

    
    
'''Exponential discounting of exploration parameter. This can be switched to be linear decay.'''
def exploration_control(eps,current_ep,total_episode,decrement,slope):
    eps -= decrement
    eps_f = m.exp(-slope*(1-eps))*eps
    if eps_f <= 0.1:
        eps_f = 0.1
    return eps, eps_f
    
    
    
def adjust_learning_rate(optimizer, initial_lr, epoch, learning_rates, decay_rate_increment):
    """
    Adjusts the learning rate using exponential decay.
    :param optimizer: PyTorch optimizer (e.g., Adam)
    :param initial_lr: Initial learning rate
    :param epoch: Current epoch number
    :param decay_rate: Rate of exponential decay (default 0.95)
    """
    # Compute new learning rate
    #new_lr = initial_lr * (decay_rate ** epoch)
    new_lr = initial_lr - epoch * decay_rate_increment
    learning_rates.append(new_lr)
    
    # Update the learning rate in the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    print(f"Learning rate adjusted to: {new_lr}")
    
    
    
    
    
def run_episode(env, 
                q_network, 
                target_q_network, 
                optimizer, 
                scheduler,
                replay_buffer, 
                reward_buffer, 
                epsilon, 
                gamma, 
                action_space, 
                district_num,
                node_num,
                update_target_freq, 
                train_freq, 
                step_count, 
                total_step_count,
                learning_rates):
    
    state, state_partition = env.reset()
    done = False
    total_reward = []
    
    while not done:
        # Select action using epsilon-greedy strategy
        action = select_action(q_network, state, state_partition, epsilon, action_space)
        
        # Perform action in the environment
        next_state, next_state_partition, reward, done, _ = env.step(action)
        
        # Store the un-normalized reward in the reward buffer
        reward_buffer.add(reward)
        
        # Normalize the reward
        reward_mean, reward_std = reward_buffer.get_reward_mean_std()
        normalized_reward = (reward - reward_mean) / (reward_std + 1e-8)
        
        # Convert the action dict into a vector
        action_vec = create_action_vector(node_num, district_num, action)

        # Store the transition in the replay buffer
        replay_buffer.add((state, action_vec, reward, next_state, done))
        
        # Update the target network periodically
        step_count += 1
        if step_count % update_target_freq == 0:
            update_target_network(q_network, target_q_network)
        
        if step_count % train_freq == 0:
            # Train the network
            train_q_network(q_network, target_q_network, optimizer, scheduler, replay_buffer, gamma, action_space, step_count)
            # Collect the current learning rate
            current_lr = scheduler.get_last_lr()[0]  # Get current learning rate (for first param group)
            learning_rates.append(current_lr)

        if step_count >= total_step_count:
            break
        
        # Update the current state
        state = next_state 
        state_partition = next_state_partition
        total_reward.append(reward)
    
    return total_reward, step_count, env.get_best_partition()