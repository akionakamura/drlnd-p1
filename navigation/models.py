import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Implements a simple fully-connected 3 layers neural network."""
    
    def __init__(self, state_size: int, action_size: int, seed: int):
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Implements a Dueling architecture."""
    
    def __init__(self, state_size: int, action_size: int, seed: int):
        super(DuelingQNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        # Dueling DQN has first common layers, then splits into two parts,
        # one for the state value and another for the action advantage.
        self.fc1 = nn.Linear(state_size, 32)
        
        self.state_value_fc1 = nn.Linear(32, 16)
        self.state_value_fc2 = nn.Linear(16, 1)
        
        self.advantage_values_fc1 = nn.Linear(32, 16)
        self.advantage_values_fc2 = nn.Linear(16, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        dense_state = torch.tanh(self.fc1(state))

        state_value = torch.tanh(self.state_value_fc1(dense_state))
        state_value = self.state_value_fc2(state_value)

        advantage_values = torch.tanh(self.advantage_values_fc1(dense_state))
        advantage_values = self.advantage_values_fc2(advantage_values)

        advantage_mean = torch.mean(advantage_values, dim=1, keepdim=True)

        action_values = state_value + advantage_values - advantage_mean

        return action_values