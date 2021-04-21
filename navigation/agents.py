from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from memory import ReplayBuffer

# Default epsilon-greedy parameters.
DEFAULT_INITIAL_EPSILON = 1.0
DEFAULT_EPSILON_DECAY = 0.997
DEFAULT_MINIMUM_EPSILON = 0.01

# Default learning parameters.
DEFAULT_DISCOUNT_FACTOR = 0.95
DEFAULT_LEARN_EVERY_STEP = 4
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_SEED = 42
DEFAULT_TAU = 1e-3


class Agent:
    """Reinforcement learning agent through Q-Learning."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        network_class: Type[nn.Module],
        double_dqn: bool = False,
        initial_epsilon: float = DEFAULT_INITIAL_EPSILON,
        epsilon_decay: float = DEFAULT_EPSILON_DECAY,
        min_epsilon: float = DEFAULT_MINIMUM_EPSILON,
        gamma: float = DEFAULT_DISCOUNT_FACTOR,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        learn_step: float = DEFAULT_LEARN_EVERY_STEP,
        tau: float = DEFAULT_TAU,
        seed: int = DEFAULT_SEED
    ):
        """Initialize the learning agent.

        Args:
            state_size (int): Size of the state space.
            action_size (int): Size of the action space.
            network_class (nn.Module): The neural network class to be used.
            double_dqn (bool): Whether or not to the use Double DQN algorithm
                during training.
            initial_epsilon (float): The initial value of epsilon for the
                epsilon-greedy algorithm.
            epsilon_decay (float): The decay factor epsilon will decrease
                at each episode.
            min_epsilon (float): The minimum value of epsilon to be used
                during training.
            gamma (float): The discount factor of expected future returns.
            learning_rate (float): The learning rate of the NN training.
            learn_step (float): The agent will update its neural network
                weights every `learn_step` steps.
            tau (float): The rate to which update the main and target networks.
            seed (float): The execution seed.

        Returns:
            Agent: The instance of the Agent.
        """
        # Initialize the main parameters.
        self.state_size = state_size
        self.action_size = action_size
        self.network_class = network_class
        self.double_dqn = double_dqn
        
        # Initialize the epsilon-greedy parameters.
        self.epsilon = self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize the learning parameters.
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learn_step = learn_step
        self.tau = tau
        self.seed = seed
        
        # Initialize both the main network and the target network.
        self.q_network = network_class(
            self.state_size, self.action_size, self.seed)
        self.q_network_target = network_class(
            self.state_size, self.action_size, self.seed)
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate)
        
        # Initialize the replay buffer.
        self.replay_buffer = ReplayBuffer()
        self.t = 0
    
    def step(
        self,
        state: np.array,
        action: int,
        reward: float,
        next_state: np.array,
        done: bool
    ):
        """Registers the agent's every step in the environment."""
        # Register in the replay buffer, and increment steps.
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.t += 1
        
        # If a multiple of `learn_step`, it is time to learn.
        if self.t % self.learn_step == 0:
            self.learn()
        
        # If episode ended, update parameters.
        if done:
            self.episode_finished()
    
    def learn(self):
        """Runs the a learning step."""
        
        # Sample transitions from the replay buffer.
        states, actions, rewards, next_states, dones = self.replay_buffer\
            .sample()
        
        # Convert everything to PyTorch tensors.
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float().squeeze()        
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(int)).float()\
            .squeeze()
        
        if self.double_dqn:
            # Double Q-learning selects the action that maximizes return from
            # the main network, but actually estimate returns from the target
            # network.
            _, action_next = self.q_network(next_states).detach().max(1)
            q_targets = self.q_network_target(next_states).detach()\
                .gather(1, action_next.unsqueeze(1)).squeeze(1)
        else:
            # Regular DQN simply fetches the target q-value from the target network,
            # selecting the action that maximizes return.
            q_targets = self.q_network_target.forward(next_states).detach().max(1)[0]
        
        # Compute the target a-values.
        q_targets = rewards + self.gamma * (q_targets * (1 - dones))
        
        # Compute the expected q-values.
        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        
        # Temporal difference error.
        td_error = F.mse_loss(q_values, q_targets)
        
        # Optimize to minimize the error.
        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()
        
        # Sync main and target networks.
        self.soft_update(self.q_network, self.q_network_target)
        
    def act(self, state: np.array):
        """Agent selects and action given an state."""
        
        # Epsilon-greedy logic, chooses and random.
        if np.random.uniform(0, 1) <= self.epsilon:
            return np.random.choice(self.action_size)
        
        # Chooses the greedy action, according with the main network.
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        action_values = self.q_network.forward(state_tensor)
        
        return np.argmax(action_values.data.numpy())
    
    def episode_finished(self):
        """At the end of an episode, update parameters"""
        # Update epsilon value.
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        
    def soft_update(self, main_model: nn.Module, target_model: nn.Module):
        """Soft update model parameters.
        
        θ_target = τ*θ_main + (1 - τ)*θ_target

        Args:
            main_model (int): weights will be copied from this model.
            target_model (int): weights will be copied to this model.
        """
        params = zip(target_model.parameters(), main_model.parameters())
        for target_param, main_param in params:
            target_param.data.copy_(
                self.tau*main_param.data + (1.0 - self.tau)*target_param.data)
