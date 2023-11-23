import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import random
import FlappyBirdEnv
# 定义深度神经网络模型
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        #state图像数组 action 1/0 reward 1 = -1 0 = 0 fail = -100 pass = 10
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))

# 定义DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=0.1, lr=0.001, buffer_capacity=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)#.unsqueeze(0)

                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def update_model(self):
        if len(self.buffer.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * (1 - dones) * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

'''
# 创建环境和Agent
env = gym.make('CartPole-v1',render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# 训练DQN
for episode in range(1000):
    state = env.reset()
    total_reward = 0
    state = state[0]

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ ,_= env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)
        total_reward += reward

        state = next_state

        agent.update_model()
        agent.update_target_network()
        env.render()
        if done:
            break

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# 在训练结束后，你可以使用训练好的模型进行测试
state = env.reset()
total_reward = 0

while True:
    env.render()
    action = agent.select_action(state)
    next_state, reward, done, _ , _= env.step(action)
    total_reward += reward

    state = next_state

    if done:
        break

print(f"Test Total Reward: {total_reward}")

env.close()
'''