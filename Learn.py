#! /usr/bin/python3

import time
import math
import gym
import numpy as np
from itertools import chain
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

class CustomCNN(BaseFeaturesExtractor):
  """
  :param observation_space: (gym.Space)
  :param features_dim: (int) Number of features extracted.
  This corresponds to the number of unit for the last layer.
  """

  def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
    super().__init__(observation_space, features_dim)
    # We assume CxHxW images (channels first)
    # Re-ordering will be done by pre-preprocessing or wrapper
    n_input_channels = observation_space.shape[0]
    self.cnn = nn.Sequential(
      nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
      nn.ReLU(),
      nn.Flatten(),
    )

    # Compute shape by doing one forward pass
    with th.no_grad():
      n_flatten = self.cnn(
      th.as_tensor(observation_space.sample()[None]).float()
    ).shape[1]

    self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

  def forward(self, observations: th.Tensor) -> th.Tensor:
    return self.linear(self.cnn(observations))

class KongmingChessEnv(gym.Env):
    def __init__(self):
        self.board = np.zeros((7, 7), dtype=np.uint8) # 0表示未翻牌,1表示焦点位置,2表示候选点,3表示翻牌.4表示不可点击区域,
        for x in chain(range(0,2), range(5, 7)):
          for y in chain(range(0, 2), range(5, 7)):
            self.board[x][y] = 4 # 不可点击区域.
        self.board[3][3] = 3
        self.focus = None
        self.candidates = None
        self.remainings = 32
        self.done = False
        self.action_space = gym.spaces.Discrete(7 * 7)
        self.observation_space = gym.spaces.Box(low=0, high=4, shape=(7, 7), dtype=np.uint8) # TODO: 是否需要修改为离散空间?

    def getFinalReward(self):
      if self.remainings == 1 and self.board[3][3] == 0:
          return math.pow(10, 9)
      else:
        return math.pow(10, 9 -  self.remainings)

    def check_over(self):
      if self.remainings == 1:
        return True

      # 找不到可以跳转的节点对
      for x in range(0, 7):
        for y in range(0, 7):
          if (x < 2 or x >= 5) and (y < 2 or y >= 5):
            continue
          if self.board[x][y] == 2 or self.board[x][y] == 3:
            continue
          if x + 2 < 7 and (self.board[x + 1][y] == 0 or self.board[x + 1][y] == 1) and (self.board[x + 2][y] == 2 or self.board[x + 2][y] == 3):
            return False
          if x - 2 >= 0 and (self.board[x - 1][y] == 0 or self.board[x - 1][y] == 1) and (self.board[x - 2][y] == 2 or self.board[x - 2][y] == 3):
            return False
          if y + 2 < 7 and (self.board[x][y + 1] == 0 or self.board[x][y + 1] == 1) and (self.board[x][y + 2] == 2 or self.board[x][y + 2] == 3):
            return False
          if y - 2 >= 0 and (self.board[x][y - 1] == 0 or self.board[x][y - 1] == 1) and (self.board[x][y - 2] == 2 or self.board[x][y - 2] == 3):
            return False
      return True


    def reset(self):
        self.board = np.zeros((7, 7), dtype=np.uint8)
        for x in chain(range(0,2), range(5, 7)):
          for y in chain(range(0, 2), range(5, 7)):
            self.board[x][y] = 4 # 不可点击区域.
        self.board[3][3] = 3
        self.focus = None
        self.candidates = None
        self.remainings = 32
        self.done = False
        return self.board

    def render(self):
      print(self.board)

    def step(self, action):
      x = action % 7
      y = action // 7
      if self.board[x][y] == 4: # 不可点击区域,惩罚
        return self.board, -100, self.done, {}
     
      if self.focus == None:
        if self.board[x][y] == 3: # 当前无焦点，却点到空格子, 惩罚
          return self.board, -100, self.done, {}

        # 逐个判断候选选点
        candidates = []
        midpoints = []
        if y - 2 >= 0 and self.board[x][y -1] == 0 and self.board[x][y - 2] == 3:
          candidates.append((x, y - 2))
          midpoints.append((x, y - 1))
        if y + 2 < 7 and self.board[x][y + 1] == 0 and self.board[x][y + 2] == 3:
          candidates.append((x, y + 2))
          midpoints.append((x, y + 1))
        if x - 2 >= 0 and self.board[x - 1][y] == 0 and self.board[x - 2][y] == 1:
          candidates.append((x - 2, y))
          midpoints.append((x - 1, y))
        if x + 2 < 7 and self.board[x + 1][y] == 0 and self.board[x + 2][y] == 1:
          candidates.append((x + 2, y))
          midpoints.append((x + 1, y))

        if len(candidates) == 0: # 点了不当的焦点，惩罚
          return self.board, -100, self.done, {}

        # 如果只有一个候选节点，则直接跳.
        if len(candidates) == 1:
          self.board[x][y] = 3
          self.board[midpoints[0][0]][midpoints[0][1]] = 3
          self.board[candidates[0][0]][candidates[0][1]] = 0
          self.remainings -= 1
          self.done = self.check_over()
          if self.done == True:
            return self.board, self.getFinalReward(), self.done, {}
          return self.board, 0, self.done, {}   

        self.focus = (x, y)
        self.candidates = candidates
        self.board[x][y] = 1
        for xx, yy in candidates:
          self.board[xx][yy] = 2
        return self.board, 0, self.done, {}

      # 已经有跳转起点了.
      # 判断点击的是否为候选节点
      if self.board[x][y] != 2:
        self.board[self.focus[0]][self.focus[1]] = 0
        self.focus = None
        for xx, yy in self.candidates:
          self.board[xx][yy] = 3
        self.candidates = None
        return self.board, -100, self.done, {}  # 点了非候选节点，惩罚

      if x == self.focus[0]:
        midX = x
        if y == self.focus[1] + 2:
          midY = y - 1
        elif y == self.focus[1] - 2:
          midY = y + 1
      elif y == self.focus[1]:
        midY = y
        if x == self.focus[0] + 2:
          midX = x - 1
        elif x == self.focus[0] - 2:
          midX = x + 1

      self.remainings -= 1
      self.board[x][y] = 0
      self.board[self.focus[0]][self.focus[1]] = 3
      self.board[midX][midY] = 3
      self.focus = None
      for xx, yy in self.candidates:
        if xx == x and yy == y:
          continue
        self.board[xx][yy] = 3
      self.candidates = None
      self.done = self.check_over()
      if self.done == True:
        return self.board, self.getFinalReward(), self.done, {}
      return self.board, 0, self.done, {}

def make_env():
        return KongmingChessEnv()

if __name__ == '__main__':
  env = make_vec_env(make_env, n_envs=4, vec_env_cls=DummyVecEnv, vec_env_kwargs={})
  #env = DummyVecEnv([lambda: env])

  policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=49),
  )
  model = PPO('MlpPolicy', env, verbose=1)
  # policy_kwargs=policy_kwargs)
  model.learn(total_timesteps=20000000)
  model.save("kongmingchess_model")

  input("模型训练结束，请按任意键开始走棋!")
  env = KongmingChessEnv()
  done = False
  obs = env.reset()
  while not done:
    # 获取动作预测
    action, _ =  model.predict(obs)
    #action = np.argmax(action)
    obs, reward, done, info = env.step(action)
    print('Reward: ', reward)
    print('Action: ', action % 7, action // 7)
    print('Observation: ')
    print(obs)
    time.sleep(1)
