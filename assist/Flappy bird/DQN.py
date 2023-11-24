import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
from torchvision import models
# 定义深度神经网络模型
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=4, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 重新计算全连接层的输入大小
        conv1_output_size = (16, 128, 72)
        conv2_output_size = (32, 32, 36)
        conv3_output_size = (16, 15, 17)
        fc1_input_size = 288

        self.fc1 = nn.Linear(fc1_input_size, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.reshape(-1, 288)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

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
        self.buffer[self.position] = (np.transpose(state, (2, 0, 1)), action, reward, np.transpose(next_state, (2, 0, 1)), done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))

# 定义DQN Agent
class DQNAgent():
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=0.1, lr=0.001, buffer_capacity=1000, batch_size=16):
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
                state = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0)
                print(state.shape)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def update_model(self):
        if len(self.buffer.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        print(states.shape)

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