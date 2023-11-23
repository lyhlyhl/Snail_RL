import FlappyBird
import gym
from gym import spaces
import pygame
import numpy as np
import random

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()

        self.action_space = spaces.Discrete(2)  # Two possible actions: 0 (do nothing) or 1 (jump)
        self.observation_space = spaces.Box(low=0, high=255, shape=(FlappyBird.SCREEN_WIDTH, FlappyBird.SCREEN_HEIGHT, 3), dtype=np.uint8)

        self.screen = pygame.display.set_mode((FlappyBird.SCREEN_WIDTH, FlappyBird.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.game = FlappyBird.Mygame()

    def reset(self):
        self.game.reset()
        return self._get_observation()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        if action == 1:
            self.game.bird.jump()

        running = self.game.step()

        # The observation, reward, done, and info are typical for OpenAI Gym environments.
        observation = self._get_observation()
        reward = self.game.reward
        done = not running
        info = {}

        return observation, reward, done, info

    def render(self):
        pygame.display.flip()
        self.clock.tick(FlappyBird.FPS)

    def close(self):
        pygame.quit()

    def _get_observation(self):
        observation = pygame.surfarray.array3d(pygame.display.get_surface())
        return np.transpose(observation, (1, 0, 2))  # Transpose to (width, height, channels) for Gym's convention

if __name__ == "__main__":
    env = FlappyBirdEnv()
    state = env.reset()

    while True:
        action = env.action_space.sample()  # Replace this with your agent's action
        next_state, reward, done, _ = env.step(action)
        env.render()

        if done:
            break

    env.close()

