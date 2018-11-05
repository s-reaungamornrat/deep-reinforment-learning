import copy
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from network import *
from ou_noise import *

class DDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, device, state_size, action_size, random_seed, hidden_in_dim, hidden_out_dim, activation, 
                 tau, lr_actor, lr_critic, weight_decay, epsilon, epsilon_decay):
             
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(DDPG, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.device = device
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Actor Network (w/ Target Network)
        self.actor_local = Network(self.state_size, self.action_size, hidden_in_dim, hidden_out_dim, activation=activation, is_actor=True).to(self.device)
        self.actor_target = Network(self.state_size, self.action_size, hidden_in_dim, hidden_out_dim, activation=activation, is_actor=True).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Network(self.state_size*2, self.action_size*2, hidden_in_dim, hidden_out_dim, activation=activation, is_actor=False).to(self.device)
        self.critic_target = Network(self.state_size*2, self.action_size*2, hidden_in_dim, hidden_out_dim, activation=activation, is_actor=False).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Same initialization
        self.__copy__(self.actor_local, self.actor_target)
        self.__copy__(self.critic_local, self.critic_target)

        # Noise process
        self.noise = OUNoise(action_size, seed=random_seed)
                
    def act(self, state, noise_scale=0.0):
        """Returns actions for given state as per current policy."""

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state.to(self.device))
        self.actor_local.train()
        return action + noise_scale*self.noise.noise()

    def target_act(self, state, noise_scale=0.0):
        """Returns actions for given state as per current policy."""

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        return self.actor_target(state.to(self.device)) + noise_scale*self.noise.noise()

    def reset(self):
        self.noise.reset()

    def update_exploration_strategy(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + ? * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= self.epsilon_decay
        self.noise.reset()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        ?_target = t*?_local + (1 - t)*?_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.-self.tau)*target_param.data)

    def __copy__(self, source, target):
        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(src_param.data)

