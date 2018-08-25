
# Project Details

The project is part of **Udacity Deep Reinforcement Learning Nano Degree program**, involving training of an agent to collect bananas in a large square world. The agent is a [**Double-Deep Q Network (DDQN)**](!https://arxiv.org/abs/1509.06461) with [**Prioritized Experience Replay**](!https://arxiv.org/abs/1511.05952). As general reinforcement learning, the problem invloves environment states percieved by the agent, actions selected and performed by the agent, rewards returned by the environment for each action at the next time step, next states. The following describe the definition of each the components  
- **Observation space** has 37 dimensions corresponding to the velocity of agent and ray-based perception of objects surrounding agent's forward direction
- **Action space** has 4 dimensions corresponding to 
   - 0 - move forward
   - 1 - move backward
   - 2 - turn left
   - 3 - turn right
- **Reward function**
   - +1 for collecting a yellow banana
   - -1 for collecting a blue banana

The problem is formed as an _episodic task_ with the goal of collecting as many bananas as possible. An episode ends when the maximum number of time steps is reached. The problem is considered **solved** if the agent attains **an average score of 13 over 100 consecutive episodes**.


## Getting Started

### Conda Environment

For those using `conda`, you can create a new `conda` environment as follows.
* _Windows_
```
conda create --name your_env_name python=3.6 
activate your_env_name 
```
* _Linux_ or _MAC_
```
conda create --name your_env_name python=3.6 
source activate your_env_name 
```

### Project Dependencies

The project requires the following libraries to be correctly installed. 

1. **Unity ML-Agents** <br>
   The installation instruction can be found at [Unity-ML website](!https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
   * Download [Unity](!https://store.unity.com/download) 
   * Clone the ML-Agents Toolkit GitHub repository 
   ```
   git clone https://github.com/Unity-Technologies/ml-agents.git
   ```
   * Download [requirements.txt](!https://github.com/Unity-Technologies/ml-agents/blob/master/python/requirements.txt) and install 
   ```
   conda install --yes --file requirements.txt
   ```
2. **Pytorch** <br>
   The pytorch version to install depends on your system configuration (e.g., operating systems, package managers, python versions, and availibility of CUDA). 
   * Select the configuration of your system at [pytorch website](!https://pytorch.org/) and follow the installation instruction described on the webpage.
   

# Instructions

The **DDQN** agent with **prioritized experience replay** is defined in Section 4 in `Navigation.ipynb`. Section 5 describes a method to train the agent. A snippet of the code to train an agent is shown below.
```
# determine torch computing device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set up environment
env = UnityEnvironment(file_name="Banana.exe")
action_size = env.brains[env.brain_names[0]].vector_action_space_size  # get dimension of action space
state_size = env.reset(train_mode=False)[env.brain_names[0]] .vector_observations.size # get dimension of state space

# initilize an DDQN agent with prioritized experience replay
agent = Agent(device, state_size, action_size)

# train the agent  
scores = dqn(env, agent, train_mode=True)
```

