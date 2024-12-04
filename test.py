import torch
import numpy as np
from dqn_agent import Agent
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# Step 1: Get env 
env = UnityEnvironment(file_name="testing/v2/DQN_NPC_Navigation_Unity.exe")
env.reset()

# Step 2: Get The Unity Environment Behavior
# Get the behavior name (formerly known as brain_name)
behavior_name = list(env.behavior_specs.keys())[0]
behavior_spec = env.behavior_specs[behavior_name]
# Step 3: Determine the size of the Action and State Spaces
# Set the number of actions (action size)
action_size = behavior_spec.action_spec.discrete_branches[0]  
print("Action size",action_size)
# Set the size of state observations (state size)
state_size = behavior_spec.observation_specs[0].shape[0]  # Assuming there is one observation, otherwise adjust index
print( "State size",state_size)

#Step 4: Initialize Agent and load trained model weights
agent = Agent(state_size=state_size, action_size=action_size)
agent.network.load_state_dict(torch.load('best_model/dqnAgent_Trained_Model_81.pth'))
agent.network.eval()


# Step 5: Interact with the environment
env.reset()
decision_steps, terminal_steps = env.get_steps(behavior_name)

while len(terminal_steps) == 0:  # Run until the episode ends
    # Get the current state
    state = decision_steps.obs[0][0]  # Assuming one agent and one observation
    # Select action using the trained agent
    action,determined = agent.act(state, eps=0.00)  # eps=0 for a fully greedy policy
    print(action)
    # Convert the action to ActionTuple and send it to the environment
    action_tuple = ActionTuple(discrete=action)
    env.set_actions(behavior_name, action_tuple)

    # Step the environment
    env.step()

    # Retrieve new states
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    if len(terminal_steps) > 0:
        print("Episode finished!")
# Step 6: Close the environment
env.close()