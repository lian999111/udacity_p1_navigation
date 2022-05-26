
# %%
from collections import deque
from dqn_agent import Agent
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment
import numpy as np

# %% Start environment
env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

# %% Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# %% Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# %% Define a DQN agent and the training process
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

def train_dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, goal_score=13):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    average_scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        # Reset the environment
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0] 
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)                          # select an action
            env_info = env.step(action)[brain_name]                 # send the action to the env
            next_state = env_info.vector_observations[0]            # get the next state
            reward = env_info.rewards[0]                            # get the reward
            done = env_info.local_done[0]                           # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        average_score = np.mean(scores_window)
        average_scores.append(average_score)
        if average_score >= goal_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'result/checkpoint.pth')
            break
    return scores, average_scores

# %% Train dqn agent
scores, average_scores = train_dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='score')
plt.plot(np.arange(len(average_scores)), average_scores, label='avg score')
plt.legend()
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('result/score.jpg')

# %% Watch a smart agent
import time
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(3):
    # Reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0] 
    score = 0
    for j in range(200):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]                 # send the action to the env
        next_state = env_info.vector_observations[0]            # get the next state
        reward = env_info.rewards[0]                            # get the reward
        done = env_info.local_done[0]                           # see if episode has finished
        state = next_state
        score += reward
        if done:
            break
        time.sleep(0.1)
    
    print('Score: {}'.format(score))

# %%
env.close()