import torch
import random
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["local_state", "global_state", "action", "reward", "next_local_state",
                                                                "next_global_state", "done"])
        self.seed = random.seed(seed)

    def add(self, local_state, global_state, action, reward, next_local_state, next_global_state, done):
        """Add a new experience to memory."""
        e = self.experience(local_state, global_state, action, reward, next_local_state, next_global_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # shape [num_agents, num_samples, state_size] --- [2, batch_size, 24]
        local_states = torch.from_numpy(np.stack([e.local_state for e in experiences if e is not None]).swapaxes(0,1)).float().to(self.device)
        # shape [num_samples, num_agentsxstate_size] --- [batch_size, 2x24=48]
        global_states = torch.from_numpy(np.stack([e.global_state for e in experiences if e is not None])).float().to(self.device)
        # shape [num_samples, num_agentsxaction_size] --- [batch_size, 2x2=4]
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).float().to(self.device)
        # shape [num_agents, num_samples, rewards_size] --- [2, batch_size, 1]
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None]).swapaxes(0,1)).float().to(self.device)
        # shape [num_agents, num_samples, state_size] --- [2, batch_size, 24]
        next_local_states = torch.from_numpy(np.stack([e.next_local_state for e in experiences if e is not None]).swapaxes(0,1)).float().to(self.device)
        # shape [num_samples, num_agentsxstate_size] --- [batch_size, 2x24=48]
        next_global_states = torch.from_numpy(np.stack([e.next_global_state for e in experiences if e is not None])).float().to(self.device)
        # shape [num_agents, num_samples, done_size] --- [2, batch_size, 1]
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None]).swapaxes(0,1)).float().to(self.device)

        return (local_states, global_states, actions, rewards, next_local_states, next_global_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)