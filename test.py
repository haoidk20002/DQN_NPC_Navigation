import torch
import time
import numpy as np
from dqn_agent import Agent
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from tqdm import tqdm

# Step 1: Set the Test Parameters
# num_episodes (int): number of test episodes

num_episodes=10             
# Step 2: Starting the environment
env = UnityEnvironment(file_name="Testing/DQN_NPC_Navigation_Unity.exe")
env.reset()

# Step 3: Get The Unity Environment Behavior
behavior_name = list(env.behavior_specs.keys())[0]
behavior_spec = env.behavior_specs[behavior_name]


# Step 4: Determine the size of the Action and State Spaces
# Set the number of actions (action size)
action_size = behavior_spec.action_spec.discrete_branches[0]  # or continuous_size depending on your environment
print("Action size",action_size)
# Set the size of state observations (state size)
state_size = behavior_spec.observation_specs[0].shape[0]  # Assuming there is one observation, otherwise adjust index


# Step 5: Initialize a DQN Agent from the Agent Class in dqn_agent.py
agent = Agent(state_size=state_size, action_size=action_size, dqn_type='DQN')

# Step 6: Load trained model weights (not finished)
agent.network.load_state_dict(torch.load('dqnAgent_Trained_Model.pth'))

# Step 7: Play the environment for specified number of episodes (not finished)
# loop from num_episodes
for i_episode in range(1, num_episodes+1):

    # reset the unity environment at the beginning of each episode
    # set train mode to false
    env_info = env.reset(train_mode=False)[brain_name]     

    # get initial state of the unity environment 
    state = env_info.vector_observations[0]

    # set the initial episode score to zero.
    score = 0

    # Run the episode loop;
    # At each loop step take an action as a function of the current state observations
    # If environment episode is done, exit loop...
    # Otherwise repeat until done == true 
    while True:
        # determine epsilon-greedy action from current sate
        action = agent.act(state)             

        # send the action to the environment and receive resultant environment information
        env_info = env.step(action)[brain_name]        

        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished

        # set new state to current state for determining next action
        state = next_state

        # Update episode score
        score += reward

        # If unity indicates that episode is done, 
        # then exit episode loop, to begin new episode
        if done:
            break

    # (Over-) Print current average score
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score), end="")

# Step 8: Close the Environment
env.close()

