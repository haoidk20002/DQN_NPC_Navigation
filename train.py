import torch
import time
import numpy as np
from dqn_agent import Agent
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from tqdm import tqdm
# Step 1: Setting parameters
# num_episodes (int): maximum number of training episodes
# epsilon (float): starting value of epsilon, for epsilon-greedy action selection
# epsilon_min (float): minimum value of epsilon
# epsilon_decay (float): multiplicative factor (per episode) for decreasing epsilon
# scores (float): list to record the scores obtained from each episode
# scores_average_window (int): the window size employed for calculating the average score (e.g. 100)
# solved_score (float): the average score required for the environment to be considered solved
# (here we set the solved_score a little higher than 13 [i.e., 14] to ensure robust learning).

num_episodes=1000
epsilon=1.0
epsilon_min=0.1
epsilon_decay=0.995
scores = []
scores_average_window = 10      
solved_score = 6000  

# Step 2: Starting the environment
env = UnityEnvironment(file_name="Testing/DQN_NPC_Navigation_Unity.exe")
env.reset()
# Step 3: Get The Unity Environment Behavior
# Get the behavior name (formerly known as brain_name)
behavior_name = list(env.behavior_specs.keys())[0]
behavior_spec = env.behavior_specs[behavior_name]
# print(behavior_name)
# print(behavior_spec)


# Step 4: Determine the size of the Action and State Spaces
# Set the number of actions (action size)
action_size = behavior_spec.action_spec.discrete_branches[0]  # or continuous_size depending on your environment
print("Action size",action_size)
# Set the size of state observations (state size)
state_size = behavior_spec.observation_specs[0].shape[0]  # Assuming there is one observation, otherwise adjust index


# Step 5: Initialize a DQN Agent from the Agent Class in dqn_agent.py
agent = Agent(state_size=state_size, action_size=action_size, dqn_type='DQN')

progress = tqdm(range(1, num_episodes+1), total= num_episodes)

# Step 6: Play the environment for specified number of episodes
for i_episode in progress:
    # Reset the environment
    # print ("Training " , i_episode)
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    # print("decision step: ", decision_steps)
    # print("terminal step: ", terminal_steps)
    # Get initial state
    state = decision_steps.obs[0][0]  # Assuming single agent and single observation space

    score = 0
    num = 0
    while True:
        # Agent takes an action (epsilon-greedy)
        action = agent.act(state, epsilon)
        # print (action)
        # Convert action into the environment's expected format (discrete action)
        # action_tuple = (np.array([action]))
        action_tuple = ActionTuple(discrete=action)
        #print("action tuple", action_tuple.shape)
        # Step the environment and receive the next state, reward, and done
        env.set_actions(behavior_name, action_tuple)
        env.step()

        # Get new environment info
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        #print("new terminal step: ",terminal_steps)
        # Check if agent is in terminal step (done)
        if len(terminal_steps) > 0:
            next_state = terminal_steps.obs[0][0]
            reward = terminal_steps.reward[0]
            done = True
        else:
            next_state = decision_steps.obs[0][0]
            reward = decision_steps.reward[0]
            done = False

        # Send (S, A, R, S') to the agent and update the agent's network
        agent.step(state, action, reward, next_state, done)

        # Update state and score
        state = next_state
        score += reward
        # print(reward)
        num += 1
        # print("num of steps: ", num)

        progress.set_description(f"reward: {reward}, step: {num}")

        if done:
            break
    
    # Record score and calculate average score over the window
    scores.append(score)
    average_score = np.mean(scores[-scores_average_window:])
    
    # Decrease epsilon for epsilon-greedy action selection
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    # Print current episode and average score
    print(fr'\rEpisode {i_episode}\tAverage Score: {average_score:.2f}\tEpsilon: {epsilon}', end="")

    # Print average score every scores_average_window episodes
    if i_episode % scores_average_window == 0:
        print(fr'\rEpisode {i_episode}\tAverage Score: {average_score:.2f}')

    # Check if the environment is solved
    if average_score >= solved_score:
        print(fr'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {average_score:.2f}')
        torch.save(agent.network.state_dict(), 'dqnAgent_Trained_Model.pth')
        break

# Step 7: Close the Environment
env.close()
