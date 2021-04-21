from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_SIZE = 1024 * 8


@dataclass
class ReplayItem:
    """Stores a state transition tuple."""
    state: np.array
    action: int
    reward: float
    next_state: np.array
    done: bool


class ReplayBuffer:
    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        self.max_size = max_size
        self.batch_size = batch_size
        
        self.buffer = deque(maxlen=max_size)
    
    def add(
        self,
        state: np.array,
        action: int,
        reward: float,
        next_state: np.array,
        done: bool
    ):
        """Add item to the buffer."""
        item = ReplayItem(state, action, reward, next_state, done)
        self.buffer.append(item)
    
    def sample(self, size: Optional[int] = None):
        """Samples `size` items from the buffer.

        Args:
            size (Optional[int]): number of items to be
                sampled. If None, uses the self.batch_size.

        Returns:
            states (List[np.array]): The list of sampled states.
            actions (List[int]): The list of sampled actions.
            rewards (List[float]): The list of sampled rewards.
            next_states (List[np.array]): The list of sampled next states.
            dones (List[bool]): The list of sampled done flags.
        """
        
        if size is None:
            size = self.batch_size
        
        # If there are less elements than the requested sample size,
        # the need to sample with replacement. Otherwise, do not replace.
        replace = len(self.buffer) < size
        
        items = np.random.choice(self.buffer, size=size, replace=replace)
        
        return tuple(zip(*[(i.state, i.action, i.reward, i.next_state, i.done)
                           for i in items]))
