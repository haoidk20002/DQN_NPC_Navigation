import torch
import numpy as np
from dqn_agent import Agent
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from tqdm import tqdm
import matplotlib.pyplot as plt
# Step 1: Setting parameters
# num_episodes (int): maximum number of training episodes
# epsilon (float): starting value of epsilon, for epsilon-greedy action selection
# epsilon_min (float): minimum value of epsilon
# epsilon_decay (float): multiplicative factor (per episode) for decreasing epsilon
# scores (float): list to record the scores obtained from each episode
# scores_average_window (int): the window size employed for calculating the average score (e.g. 100)
# solved_score (float): the average score required for the environment to be considered solved

num_episodes=1000
epsilon=0.5
epsilon_min=0.1
epsilon_decay=0.8
scores = []
scores_average_window = 5     
solved_score = 5000
decision_step_interval = 5 
# 4400 + 850 = 5250 => Avg approx 5250

# Step 2: Starting the environment
env = UnityEnvironment(file_name="Testing/DQN_NPC_Navigation_Unity.exe")
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
agent = Agent(state_size=state_size, action_size=action_size)

progress = tqdm(range(1, num_episodes+1), total= num_episodes)

# Step 6: Play the environment for specified number of episodes
try:
    for i_episode in progress:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # Get initial state
        state = decision_steps.obs[0][0]
        score = 0
        accumulated_reward = 0
        step_count = 0
        while True: # Loop frequency: every decision (decision every 5 steps)
            # Agent takes an action (epsilon-greedy)
            action = agent.act(state, epsilon)
            # Convert action into the environment's expected format (discrete action)
            action_tuple = ActionTuple(discrete=action)

            # Step the environment and receive the next state, reward, and done
            env.set_actions(behavior_name, action_tuple)
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

            # Accumulate reward
            accumulated_reward += reward
            score += reward
            if score < 0:
                # Clamp the score to be within the range [0, 100000000]
                score = max(0, min(score, 100000000))
            step_count += 1
            #print("Step: ",step_count)
            progress.set_description(f"Reward: {reward:.2f}, Step: {step_count}, Score: {score:.2f}")
            # Update state and score every decision_step_interval steps
            if step_count % decision_step_interval == 0 or done:
                # Send (S, A, R, S') to the agent and update the agent's network
                agent.step(state, action, accumulated_reward, next_state, done)
                # Update state and score
                state = next_state
                accumulated_reward = 0  # Reset accumulated reward


            # Check if the episode is done
            if done:
                break
        
        # Record score and calculate average score over the window
        scores.append(score)
        average_score = np.mean(scores[-scores_average_window:])
        max_score = max(scores)
        # Decrease epsilon for epsilon-greedy action selection
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # Print current episode and average score
        print(fr' Episode {i_episode} Score: {score:.2f} Average Score: {average_score:.2f} Max Score: {max_score:.2f} Epsilon: {epsilon}')

        # Print average score every scores_average_window episodes
        if i_episode % scores_average_window == 0:
            print(fr' Episode {i_episode} Average Score: {average_score:.2f}')

        # Check if the environment is solved
        if average_score >= solved_score:
            print(fr'Environment solved in {i_episode} episodes! Average Score: {average_score:.2f}, Max Score: {max_score:.2f}')
            torch.save(agent.network.state_dict(), 'dqnAgent_Trained_Model.pth')
            break
except KeyboardInterrupt:
    print("Training interrupted by user.")
finally:
    # After the training loop
    plt.plot(scores)
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.title('Scores over Episodes')
    plt.savefig('scores_plot.png')
    plt.show()
    print("Episode Passed: ", i_episode , "Max Score: ", max_score)

    # Step 7: Close the Environment
    env.close()
