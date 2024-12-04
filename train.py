import torch
import numpy as np
from dqn_agent import Agent
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# Step 1: Setting parameters

# num_episodes (int): maximum number of training episodes
# epsilon (float): starting value of epsilon, for epsilon-greedy action selection
# epsilon_min (float): minimum value of epsilon
# epsilon_decay (float): multiplicative factor (per episode) for decreasing epsilon
# scores (float): list to record the scores obtained from each episode
# scores_average_window (int): the window size employed for calculating the average score (e.g. 100)
# solved_score (float): the average score required for the environment to be considered solved
num_episodes=20
epsilon=0.5
epsilon_min=0.1
epsilon_decay=0.99
scores = []    

# Step 2: Starting the environment
env = UnityEnvironment(file_name="testing/v2/DQN_NPC_Navigation_Unity.exe")
env.reset()

# Step 3: Get The Unity Environment Behavior
# Get the behavior name (formerly known as brain_name)
behavior_name = list(env.behavior_specs.keys())[0]
behavior_spec = env.behavior_specs[behavior_name]

# Step 4: Determine the size of the Action and State Spaces
# Set the number of actions (action size)
action_size = behavior_spec.action_spec.discrete_branches[0]  
print("Action size",action_size)
# Set the size of state observations (state size)
state_size = behavior_spec.observation_specs[0].shape[0]  # Assuming there is one observation, otherwise adjust index
print( "State size",state_size)

# Step 5: Initialize a DQN Agent from the Agent Class in dqn_agent.py
agent = Agent(state_size=state_size, action_size=action_size, learning_rate=1e-4, replay_memory_size=1e5, batch_size=16, gamma=0.8, target_tau=1e-3, update_rate=5, seed=0)
# If you want to load a trained model, uncomment the following line
agent.network.load_state_dict(torch.load('best_model/dqnAgent_Trained_Model_81.pth')) # At ep 81 stops at ep 0.5
start_episode = 81
progress = tqdm(range(start_episode, num_episodes+start_episode), total= num_episodes)

# Step 6: Play the environment for specified number of episodes
for i_episode in progress:
    # Wait for the Unity environment to reset itself
    env.reset()
    # Get the initial state
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    state = decision_steps.obs[0][0]
    # Reset the score
    total_reward = list()
    reached_goal = False
    done = False
    action = 0
    score = 0
    step_count = 0
    while True: # Loop frequency: every step 
        # Agent takes an action (epsilon-greedy)
        action, determined = agent.act(state, epsilon)
        # Convert action into the environment's expected format (discrete action)
        # print(action)
        action_tuple = ActionTuple(discrete=action)
        # Step the environment and receive the next state, reward, and done
        env.set_actions(behavior_name, action_tuple)
        step_count += 1 
        env.step()
        # Get new environment info
        decision_steps, terminal_steps = env.get_steps(behavior_name) 
        # Check if agent is in terminal step (done)
        if len(terminal_steps) > 0:
            next_state = terminal_steps.obs[0][0]
            reward = terminal_steps.reward[0]
            done = True
        else:
            next_state = decision_steps.obs[0][0]
            reward = decision_steps.reward[0]
            done = False
        score += reward
        progress.set_description(f"Reward: {reward:.2f}, Step: {step_count}, Score: {score:.2f}, Epsilon: {epsilon:.2f}")
        # Send (S, A, R, S') to the agent and update the agent's network
        agent.step(state, action, reward, next_state, done)
        # Update state and score
        state = next_state
        total_reward.append(reward)
        if len(total_reward) > 100:
            if sum(total_reward[-100:]) == 0:
                done = True
        # Check if the episode is done
        if reward == 1000:
            reached_goal = True
        if done:
            break
    # Append the score to the scores list   
    scores.append(score)
    # Decrease epsilon for epsilon-greedy action selection
    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    # Save model every 5 episodes
    if reached_goal == True or i_episode % 5 == 0:
        torch.save(agent.network.state_dict(), f'saved_model/dqnAgent_Trained_Model_{i_episode}.pth')

# Step 7: Close the Environment
env.close()
# Step 8: Draw the line chart of the scores over the episodes
plt.figure(figsize=(10, 5))
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Score over Episodes')
# Set the x and y axes to use integer-only locators
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# Save the plot as an image file
plt.savefig('result/training_scores.png')
