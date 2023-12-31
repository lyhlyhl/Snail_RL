import DQN
import FlappyBirdEnv
import torch
env = FlappyBirdEnv.FlappyBirdEnv()
state = env.reset()
state_size = env.observation_space.shape[2]
print(env.observation_space.shape)
action_size = env.action_space.n
agent = DQN.DQNAgent(state_size, action_size)

for episode in range(10000):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _  = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)
        total_reward += reward

        state = next_state

        agent.update_model()

        env.render()
        if done:
            break
    agent.update_target_network()
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# 在训练结束后，你可以使用训练好的模型进行测试
state = env.reset()
total_reward = 0

while True:
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    env.render()
    total_reward += reward
    state = next_state

    if done:
        break

print(f"Test Total Reward: {total_reward}")
torch.save(agent.target_net.state_dict(), 'dqn_model.pth')
env.close()