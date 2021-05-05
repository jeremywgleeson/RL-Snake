# %matplotlib inline
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gym
from PIL import Image
from itertools import count

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display


class DQN(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super().__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.out = nn.Linear(self.fc2_dims, self.n_actions)

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

class Agent():
    def __init__(self, strategy, num_actions, device):
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    # returns int, not tensor
    def select_action(self, observation, policy_net, ep_num):
        rate = self.strategy.get_exploration_rate(ep_num)

        if np.random.random() < rate:
            action = np.random.randint(self.num_actions)
        else:
            state = T.tensor([observation]).to(self.device)
            with T.no_grad():
                actions = policy_net(state)
                action = T.argmax(actions).item()
        return action
        # return T.tensor([action]).to(self.device)

class ReplayMemory():
    def __init__(self, capacity, input_dims, device):
        self.capacity = capacity

        self.state_memory = np.zeros((self.capacity, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.capacity, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.capacity, dtype=np.int32)
        self.reward_memory = np.zeros(self.capacity, dtype=np.float32)
        self.terminal_memory = np.zeros(self.capacity, dtype=np.bool)

        self.device = device

        self.mem_count = 0

    def push(self, state, action, reward, new_state, done):
        index = self.mem_count%self.capacity
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done

        self.mem_count += 1


    # returns (state_batch, new_state_batch, reward_batch, terminal_batch, action_batch)
    # all of type tensor EXCEPT action_batch of type np.array
    def sample(self, batch_size):
        batch = np.random.choice(self.capacity, batch_size, replace=False)

        state_batch = T.tensor(self.state_memory[batch]).to(self.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.device)
        action_batch = self.action_memory[batch]

        return state_batch, new_state_batch, reward_batch, terminal_batch, action_batch

    def can_provide_sample(self, batch_size):
        return self.mem_count >= batch_size

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            np.exp(-1. * current_step * self.decay)

class CartPoleEnvManager():
    def __init__(self, device, size = (20,20)):
        self.device = device
        self.env = self.env = gym.make('CartPole-v0')
        self.done = False
        # self.actions_list = [(0,1), (0,-1), (1,0), (-1,0)]

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        next_state, reward, done, _ = self.env.step(action)
        self.done = done
        return next_state, reward, done

class QValues():
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions, batch_index):
        results = policy_net(states)
        return results[batch_index, actions]
        # return results.gather(dim=0, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, new_states, terminals):
        q_next = target_net(new_states)
        q_next[terminals] = 0.0
        return T.max(q_next, dim=1)[0]
        # final_state_locations = next_states.flatten(start_dim=1) \
        #     .max(dim=1)[0].eq(-1).type(T.bool)
        # non_final_state_locations = (final_state_locations == False)
        # non_final_states = next_states[non_final_state_locations]
        # non_final_states = next_states
        # batch_size = next_states.shape[0]
        # values = T.zeros(batch_size).to(QValues.device)
        # values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        # return values
        # return target_net(next_states).max(dim=1)[0].detach()

def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n", \
        moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = T.tensor(values, dtype=T.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = T.cat((T.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = T.zeros(len(values))
        return moving_avg.numpy()

batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 10000
max_steps = 500
input_dims = [40]

device = T.device("cuda" if T.cuda.is_available() else "cpu")

em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size, input_dims, device)

policy_net = DQN(input_dims, 22, 22, em.num_actions_available()).to(device)
target_net = DQN(input_dims, 22, 22, em.num_actions_available()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)


episode_rewards = []
for episode in range(num_episodes):
    tot_reward = 0
    state = em.reset()
    em.render()
    step = 0
    while step < max_steps:
        # print("state ", state)
        action = agent.select_action(state, policy_net, episode)
        # print("action", action)
        next_state, reward, done = em.take_action(action)
        em.render()
        # print(new_exp)
        memory.push(state, action, reward, next_state, done)
        
        state = next_state
        tot_reward += reward
        step += 1
        if reward > 1:
            step = 0
        # actually train here?
        if memory.can_provide_sample(batch_size):
            states, new_states, rewards, terminals, actions = memory.sample(batch_size)
            batch_index = np.arange(batch_size, dtype=np.int32)

            current_q_values = QValues.get_current(policy_net, states, actions, batch_index)
            next_q_values = QValues.get_next(target_net, new_states, terminals)

            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(target_q_values, current_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if em.done:
            break

    episode_rewards.append(tot_reward)
    print("reward:",tot_reward, "\trate:", agent.strategy.get_exploration_rate(episode))
    plot(episode_rewards, 100)
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

# em.close()
