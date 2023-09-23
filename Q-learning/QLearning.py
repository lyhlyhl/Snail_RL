import gymnasium as gym
import numpy as np
import time

#Qlearning 智能体
class QLearning():
    def __init__(self,obs_n , act_n, learning_rate = 0.1, gamma = 0.9, egreed = 0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = egreed
        self.Q = np.zeros((obs_n,act_n))
    # 贪婪策略
    def sample(self, obs):
        if np.random.uniform(0,1) < 1.0 - self.epsilon:
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action
    # Q表学习
    def learning(self, obs, act, reward, next_obs, done):
        predictQ = self.Q[obs,act]
        if done:
            targetQ = reward
        else:
            maxQ = np.max(self.Q[next_obs, :])
            targetQ = reward + self.gamma*maxQ
        self.Q[obs, act] += self.lr * (targetQ - predictQ)

    def predict(self, obs):
        Qlist = self.Q[obs,:]
        maxQ = np.max(Qlist)
        action_list = np.where(Qlist == maxQ)[0]
        action = np.random.choice(action_list)

        return action
def run_episode(env, agent, render = False):
    total_steps = 0
    total_reward = 0

    obs = env.reset()[0]
    while True:
        action = agent.sample(obs)
        obs_next, reward, done, _, _ = env.step(action)
        agent.learning(obs, action, reward, obs_next, done)
        obs = obs_next
        total_reward += reward
        total_steps += 1

        if render:
            env.render()
        if done:
            break
    return total_reward, total_steps

def tes_episode(env, agent):
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

def main():
   env = gym.make("CliffWalking-v0", render_mode=None)
   observation, info = env.reset(seed=42)
   agent = QLearning(
           obs_n=env.observation_space.n,
           act_n=env.action_space.n,
           learning_rate=0.1,
           gamma=0.9,
           egreed=0.1)
   for episode in range(300):
      ep_reward, ep_steps = run_episode(env, agent, False)
      print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))
   env = gym.make("CliffWalking-v0", render_mode="human")
   observation, info = env.reset(seed=42)
   tes_episode(env, agent)

if __name__ == '__main__':
    main()