import gymnasium as gym
import numpy as np
import time
def main():
   env = gym.make("FrozenLake-v1", render_mode=None)
   observation, info = env.reset(seed=42)
   agent = SARSA_Agent(
           obs_n=env.observation_space.n,
           act_n=env.action_space.n,
           learning_rate=0.1,
           gamma=0.9,
           e_greed=0.1)
   for episode in range(10000):
      #action = env.action_space.sample()  # this is where you would insert your policy
      #observation, reward, terminated, truncated, info = env.step(action)
      ep_reward, ep_steps = run_episode(env, agent, False)
      print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))
      #if terminated or truncated:
      #observation, info = env.reset()
   env = gym.make("FrozenLake-v1", render_mode="human")
   observation, info = env.reset(seed=42)
   test_episode(env, agent)


class SARSA_Agent(): #SARAS 策略
   def __init__(self, obs_n, act_n, learning_rate = 0.01, gamma = 0.9, e_greed = 0.1):
      self.act_n = act_n
      self.lr = learning_rate
      self.gamma = gamma
      self.epslion = e_greed
      self.Q = np.zeros((obs_n, act_n))

   def sample(self, obs):
      if np.random.uniform(0, 1) < 1.0 - self.epslion:
         action = self.predict(obs)
      else:
         action = np.random.choice(self.act_n)

      return action

   # 预测下一个动作
   def predict(self, obs):
      Qlist = self.Q[obs, :]
      maxQ = np.max(Qlist)
      action_list = np.where(Qlist == maxQ)[0]
      action = np.random.choice(action_list)

      return action

   # 学习Q表格
   def learning(self,obs ,action, reward, next_obs, next_action, done):
      predictQ = self.Q[obs, action]
      if done:
         target_Q = reward
      else:
         target_Q = reward + self.gamma * self.Q[next_obs, next_action]
      self.Q[obs, action] += self.lr * (target_Q - predictQ)

   # 保存Q表格数据到文件
   def save(self):
      npy_file = './q_table.npy'
      np.save(npy_file, self.Q)
      print(npy_file + ' saved.')

      # 从文件中读取Q值到Q表格中

   def restore(self, npy_file='./q_table.npy'):
      self.Q = np.load(npy_file)
      print(npy_file + ' loaded.')

def run_episode(env, agent, render = False):
   total_steps = 0
   total_reward = 0

   obs = env.reset()[0]
   action = agent.sample(obs)
   while True:
      obs_next, reward, done ,_ ,_= env.step(action)
      next_action = agent.sample(obs_next)
      agent.learning(obs, action, reward, obs_next, next_action, done)
      action = next_action
      obs = obs_next
      total_reward += reward
      total_steps += 1

      if render:
         env.render()
      if done:
         break
   return total_reward, total_steps

def test_episode(env, agent):
   total_reward = 0
   obs, infor = env.reset()
   while True:
      action = agent.predict(obs)  # greedy
      next_obs, reward, done, _, _ = env.step(action)
      total_reward += reward
      obs = next_obs
      time.sleep(0.5)
      env.render()
      if done:
         break
   return total_reward

if __name__ == '__main__':
    main()