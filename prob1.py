

# Load packages
import numpy as np
import gym
import torch

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
from DQN_network import ExperienceReplayBuffer, Network
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def fill_buffer(env, n):
    state = env.reset()
    
    for _ in range(n):
        action = env.action_space.sample()
        next_state, reward, done, _  = env.step(action)
        exp = state, action, reward, next_state, done
        buffer.append(exp)
        if done:
            state = env.reset()
        else:
            state = next_state


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
n = len(env.observation_space.high)
m = env.action_space.n
env.reset()
buffer_len = 30000
buffer = ExperienceReplayBuffer(max_len=buffer_len)


# Parameters
N_episodes = 500                             # Number of episodes
discount_factor = 0.95                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
epsilon = 0.1
N = 128
C = int(buffer_len/N)
step = 0
# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode
render_env = False
# Random agent initialization
#agent = RandomAgent(n_actions)
agent = Network(n, m, 64)
target_net = copy.deepcopy(agent)
optimizer = optim.Adam(agent.parameters(), lr=0.0001)
### Training process

print("filling buffer")
fill_buffer(env, int(buffer_len/2))
# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:
        if render_env:
            env.render()
        state_tensor = torch.tensor(np.array([state]), requires_grad=False, dtype=torch.float32)

        actions = agent.forward(state_tensor)
        

        if np.random.rand() > epsilon:
            action  = actions.max(1)[1].item()
        else:
            action = env.action_space.sample()

        next_state, reward, done, _  = env.step(action)

        exp = state, action, reward, next_state, done
        buffer.append(exp)
        # Update episode reward
        
        states, actions, rewards, next_states, dones = buffer.sample_batch(N)
        next_states_tensor = torch.tensor(np.array(next_states), requires_grad=False, dtype=torch.float32)
        optimizer.zero_grad()

        states = torch.tensor(np.array(states), requires_grad=True, dtype=torch.float32)
         
        targets = target_net.forward(next_states_tensor)
        y = torch.tensor(rewards, dtype=torch.float32) + (1 - torch.tensor(dones, dtype=torch.int32)) * discount_factor * targets.max(1)[0]
        y = y.view(-1, 1)
        val = agent.forward(states)
        
        val = val.gather(1, torch.tensor(actions).view(-1, 1))

        loss = nn.functional.mse_loss(val, y)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1)
        optimizer.step()

        total_episode_reward += reward
        # Update state for next iteration
        state = next_state
        t+= 1
        step+= 1

        if step % C == 0:
            target_net = copy.deepcopy(agent)

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
