import math
import torch
import numpy as np
from collections import namedtuple


class SumTree(object):
    '''
     code apdated from https://raw.githubusercontent.com/takoika/PrioritizedExperienceReplay/master/sum_tree.py
    '''
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = math.ceil(math.log(max_size+1, 2))+1
        self.tree_size = 2**self.tree_level-1
        self.tree = [0. for i in range(self.tree_size)]
        self.data = [None for i in range(self.max_size)]
        self.size = 0
        self.cursor = 0

    def add(self, contents, value):
        index = self.cursor
        self.cursor = (self.cursor+1)%self.max_size
        self.size = min(self.size+1, self.max_size)

        self.data[index] = contents
        self.val_update(index, value)

    def get_val(self, index):
        tree_index = 2**(self.tree_level-1)-1+index
        return self.tree[tree_index]

    def val_update(self, index, value):
        tree_index = 2**(self.tree_level-1)-1+index
        diff = value-self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex-1)/2)
            self.reconstruct(tindex, diff)
    
    def find(self, value, norm=True):
        '''
           value: relative of the sum at the root (0-1)
        '''
        #print('In find value, root value, tree level ', value, self.tree[0], self.tree_level)
        if isinstance(self.tree[0], complex): raise ValueError('root of tree in find is complex ', self.tree[0])
        if isinstance(value, complex): raise ValueError('value in find is complex ', value)
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):
        '''
            return 
              value of data
              value stored in tree which in this case the priority
              index to data
        '''
        if 2**(self.tree_level-1)-1 <= index:
            #print(self.tree_level, 2**(self.tree_level-1)-1)
            return self.data[index-(2**(self.tree_level-1)-1)], self.tree[index], index-(2**(self.tree_level-1)-1)

        left = self.tree[2*index+1]
        #print('In _find, left, value ', left, value)

        if value <= left:
            return self._find(value,2*index+1)
        else:
            return self._find(value-left,2*(index+1))
        
    def print_tree(self):
        for k in range(1, self.tree_level+1):
            for j in range(2**(k-1)-1, 2**k-1):
                print(self.tree[j], end=' ')
            print()

    def filled_size(self):
        return self.size
        
class ReplayMemory(object):
    """ 

    see https://arxiv.org/pdf/1511.05952.pdf .

    """
    
    def __init__(self, memory_size, batch_size, alpha, seed, device):
        """
        Parameters
        ----------
        memory_size (int): sample size to be stored
        batch_size (int): batch size to be selected by `select` method
        alpha (float): determine the degree of priorization used in sampling the buffer: 
            0 for uniform random, 
            1 for pure priority based selection
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        seed (int): initialization value for random generators
        device (string): pytorch-based computation device: cpu or cuda 
        """
        np.random.seed(seed)
        
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.alpha = alpha
        self.device = device

    def add(self, state, action, reward, next_state, done, priority):
        """
        Add a new experience.
        
        Parameters
        ----------
        state (37-element float list): current state of environment (at time t)
        action (int): action applied at time t
        reward (float): reward returned by the environment at time t+1
        next_state (37-element float list): next state of environment (at time t+1)
        done (bool): whether the episode ends
        priority (float): priority of this experience (state, action, reward, next_state, done)
        """
        e = self.experience(state, action, reward, next_state, float(np.uint8(done)))
        if isinstance(priority, complex): raise ValueError('In ReplayMem::add, Priority is complex ', priority)
        if isinstance(self.alpha, complex): raise ValueError('In ReplayMem::add, self.alpha is complex ', self.alpha)
        if isinstance(priority**self.alpha, complex): 
            raise ValueError('In ReplayMem::add, priority {} ** self.alpha {} = {} is complex '.format(priority,
                                                    self.alpha, priority**self.alpha))
                                                                       
        self.tree.add(e, float(priority**self.alpha))

    def select(self, beta):
        """ 
        Randomly sampling the experiences in the replay memory: the degree of randomness is determined by self.alpha.
        The degree/strength of important sampling weights is determined by beta
        
        Parameters
        ----------
        beta (float) : strength of correction (annealing) for non-uniform sampling, default 0.5
                       value should gradually increase to 1. at the end of learning (the q-value converges)
        
        Returns
        -------
        states (floating tensor): torch tensor of size [batch_size, 37]
        actions (long tensor): torch tensor of size [batch_size, 1]
        rewards (floating tensor): : torch tensor of size [batch_size, 1]
        next_states (floating tensor): torch tensor of size [batch_size, 37]
        dones (floating tensor): : torch tensor of size [batch_size, 1]
        weights (floating tensor): : torch tensor of size [batch_size, 1] for important sampling weights
        indices (list): indices of sample sampled from the replay memory (i.e., sample positions in a sum tree)
        """
        
        if self.tree.filled_size() < self.batch_size:
            return None, None, None, None, None, None, None

        states = [None]*self.batch_size
        actions = [None]*self.batch_size
        rewards = [None]*self.batch_size
        next_states = [None]*self.batch_size
        dones = [None]*self.batch_size
        
        indices = [None]*self.batch_size
        weights = np.zeros(self.batch_size, dtype=np.float32)
        priorities = [None]*self.batch_size
        for bidx in range(self.batch_size):
            r = np.random.uniform()
            data, priority, index = self.tree.find(r)
            priorities[bidx] = priority
            weights[bidx] = (1./self.tree.filled_size()/priority)**beta if priority > 1e-16 else 0
            indices[bidx] = index
            states[bidx] = data.state
            actions[bidx] = data.action
            rewards[bidx] = data.reward
            next_states[bidx] = data.next_state
            dones[bidx] = data.done
            self.priority_update([index], [0]) # To avoid duplicating
            
        
        self.priority_update(indices, priorities) # Revert priorities
        if not np.isclose(np.amax(weights), 0.):
            weights /= np.amax(weights) # Normalize for stability
        
        # convert to torch 
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones)).float().to(self.device)
        weights = torch.from_numpy(np.vstack(weights)).float().to(self.device)
        
        return states, actions, rewards, next_states, dones, weights, indices

    def priority_update(self, indices, priorities):
        """ 
        Update priorities of samples already stored in the replay memory.
        
        Parameters
        ----------
        indices (list): indices of sample sampled from the replay memory (i.e., sample positions in a sum tree)
        """
        for i, p in zip(indices, priorities): self.tree.val_update(i, float(p**self.alpha))
    

    def __len__(self):
        """Size of the replay memory."""
        return self.tree.filled_size()
    