import torch
import numpy as np

from ddpg import *

class MADDPG:
    def __init__(self, device, seed, gamma, ddpg_settings):
    
        '''
            ddpg_settings: dict 
        '''
        super(MADDPG, self).__init__()
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.device = device
        self.marl = [DDPG(device=device, **ddpg_settings), DDPG(device=device, **ddpg_settings)]
        self.gamma = gamma
    
    def reset(self):
        for agent in self.marl:
            agent.reset()
    def act(self, obs_per_agent, noise_scale=0.0):
        """get actions from all agents in the MADDPG object
            obs_per_agent: numpy of shape 2x24 where 2 is num agnets and 24 is state size
        """
        # torch require input of shape [num_smaple, state_size] so we unsqueeze the 1st dim
        actions = [agent.act(obs[np.newaxis,:], noise_scale) for agent, obs in zip(self.marl, obs_per_agent)]
        return actions
    
    def critic_loss_function(self, agent_number, global_states, actions, rewards, dones, next_local_states, next_global_states):
        '''
            agent : a selected ddpg agent
        '''
        
        next_target_actions = [a.target_act(next_local_state) for a, next_local_state in zip(self.marl, next_local_states)]
        #print('next_target_actions ', len(next_target_actions), next_target_actions[0].shape)
        next_target_actions = torch.cat(next_target_actions, dim=-1)
        #print('next_target_actions ', next_target_actions.shape)
        with torch.no_grad():
            q_next = self.marl[agent_number].critic_target(next_global_states, next_target_actions.to(self.device))
        #print('q_next ', q_next.shape)
        #print('reward[agent_number] ', rewards[agent_number].shape, '  done[agent_number] ',  dones[agent_number].shape)
        y = rewards[agent_number] + self.gamma * q_next * (1. - dones[agent_number])
        #print('y ', y.shape)
        #print('global states ', global_states.shape, ' actions ', actions.shape)
        q = self.marl[agent_number].critic_local(global_states, actions)
        #print('q ', q.shape)
        return F.smooth_l1_loss(q, y.detach())
    
    def actor_loss_function(self, agent_number, local_states, global_states):
        
        predicted_actions = [self.marl[i].actor_local(state) if i == agent_number \
                             else self.marl[i].actor_local(state).detach() \
                             for i, state in enumerate(local_states)]
        #print('predicted_actions ', len(predicted_actions), predicted_actions[0].shape)
        predicted_actions = torch.cat(predicted_actions, dim=-1)
        #print('predicted_actions ', predicted_actions.shape)
        return -self.marl[agent_number].critic_local(global_states, predicted_actions).mean()
    
    def update(self, experiences):
        '''
            update both learned and target actor and critic networks in all ddpg agents
        '''
        for agent_number in range(len(self.marl)):
            self.update_local(experiences, agent_number)
        self.update_targets() # soft update target network
        self.update_exploration_strategy()
    
    def update_exploration_strategy(self):
        for agent in self.marl:
            agent.update_exploration_strategy()
            
    def update_local(self, experiences, agent_number):
        '''
            Update learned critic and actor networks
            experiences: random samples from replay buffer
            agent_number: index to an agent
        '''
        local_states, global_states, actions, rewards, next_local_states, next_global_states, dones = experiences
        # get selected agent
        agent = self.marl[agent_number]
        
        #----------------update critic-----------------------
        critic_loss = self.critic_loss_function(agent_number, global_states, actions, rewards, dones, next_local_states, next_global_states)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1.)
        agent.critic_optimizer.step()
        
        #----------------update actor-----------------------
        actor_loss = self.actor_loss_function(agent_number, local_states, global_states)
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()
        
        return critic_loss.cpu().detach().item(), actor_loss.cpu().detach().item()
    
    def update_targets(self):
        '''Update target critic and actor networks'''
        for agent in self.marl:
            agent.soft_update(agent.critic_local, agent.critic_target)
            agent.soft_update(agent.actor_local, agent.actor_target)