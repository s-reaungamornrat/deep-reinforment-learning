
# Project Details

The project is part of Udacity Deep Reinforcement Learning Nano Degree program, involving training of 20 double-jointed arms to move to target locations. The arms are represented by actor-critic neural networks trained using a [DDPG](https://arxiv.org/abs/1509.02971) framework. The problem is episodic and invloves environment states percieved by the DDPG agents, **continuous actions** selected and performed by the agents, rewards returned from the environment for each action at the next time step, and transition to next states. <br>

**Observation space** has 33 dimensions corresponding to position, rotation, velocity, and angular velocities of the arms. <br>
**Action space** has 4 dimensions corresponding to torque applicable to two joints. <br>
**Reward** is +0.1 for each time step that the agents' hands are at target locations, so the goal of the agents is to maintain their positions at target locations for as many time steps as possible. The problem is considered solved if the agents attains an average score of 30 over 100 consecutive episodes.<br>

# Getting Started

## Conda Environment
For those using conda, you can create a new conda environment as follows.

+ Windows
```
conda create --name your_env_name python=3.6 
activate your_env_name 
```
+ Linux or MAC
```
conda create --name your_env_name python=3.6 
source activate your_env_name 
```

## Project Dependencies

The project requires the following libraries to be correctly installed.

1. Unity ML-Agents 
The installation instruction can be found at Unity-ML website
    - Download Unity
    - Clone the ML-Agents Toolkit GitHub repository
    ```
        git clone https://github.com/Unity-Technologies/ml-agents.git
    ```
    - Download requirements.txt and install
    ```
        conda install --yes --file requirements.txt
    ```
2. Pytorch 
The pytorch version to install depends on your system configuration (e.g., operating systems, package managers, python versions, and availibility of CUDA).
    - Select the configuration of your system at pytorch website and follow the installation instruction described on the webpage.
    
# Instructions

A DDPG agent is based on an actor network and a critic network, both defined in `model.py`. The agent itself is described in `ddpg_agent.py`. Section 4 of `Continuous_Control.ipynb` describes an approach to train DDPG agents. A snippet of the code to train an agent is shown below.

```
# determine torch computing device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set up environment
env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64') # path to a Unity Reacher environment
action_size = env.brains[env.brain_names[0]].vector_action_space_size  # get dimension of action space
state_size = env.reset(train_mode=False)[env.brain_names[0]].vector_observations.size # get dimension of state space

# agent configuration
settings = {'buffer_size':int(1e6), 'batch_size':256, 'gamma':0.99, 'tau':1e-3, 'lr_actor':1e-3, 'lr_critic':1e-3,
           'weight_decay':0., 'epsilon':1., 'epsilon_decay':1e-6, 'num_batch_permute':10}
           
# initilize an DDPG agent 
agent = Agent(device, state_size, action_size, random_seed, **settings)

# train the agent  
scores = ddpg()
```
