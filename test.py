import torch
import numpy as np
from dqn_agent import Agent
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from tqdm import tqdm
import matplotlib.pyplot as plt

# Step 1: Set the Test Parameters
num_episodes = 100
model_path = 'dqnAgent_Trained_Model.pth'

# Step 2: Starting the environment
try:
    env = UnityEnvironment(file_name="Testing/DQN_NPC_Navigation_Unity.exe")
    env.reset()
except Exception as e:
    print(f"Error initializing Unity environment: {e}")
    exit()

# Step 3: Get The Unity Environment Behavior
behavior_name = list(env.behavior_specs.keys())[0]
behavior_spec = env.behavior_specs[behavior_name]

# Step 4: Determine the size of the Action and State Spaces
action_size = behavior_spec.action_spec.discrete_branches[0]
print("Action size", action_size)
state_size = behavior_spec.observation_specs[0].shape[0]
print("State size", state_size)

# Step 5: Initialize a DQN Agent from the Agent Class in dqn_agent.py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = Agent(state_size=state_size, action_size=action_size, device=device)

# Step 6: Load the trained model
agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=device))
agent.qnetwork_local.eval()

# Step 7: Run the environment using the trained model
scores = []
progress = tqdm(range(1, num_episodes + 1), total=num_episodes)

for i_episode in progress:
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    state = decision_steps.obs[0][0]

    score = 0
    while True:
        # Move state to device
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Agent takes an action using the trained model (no epsilon-greedy)
        action = agent.act(state, epsilon=0.0)

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

        # Move next_state to device
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)

        # Update state and score
        state = next_state
        score += reward

        if done:
            break

    scores.append(score)
    average_score = np.mean(scores[-5:])  # Calculate average score over the last 5 episodes

    # Print current episode and average score
    print(f'Episode {i_episode} Score: {score:.2f} Average Score: {average_score:.2f}')

# Step 8: Close the Environment
env.close()

# Print final scores and average score
print("Final Scores:", scores)
print("Final Average Score:", np.mean(scores))

# Plot the scores
plt.plot(scores)
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.title('Scores over Episodes')
plt.savefig('test_scores_plot.png')
plt.show()