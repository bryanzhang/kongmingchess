#! /usr/bin/python3

import random
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

    hidden_dim = 512
    self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
#    self.linear = nn.Sequential(
#        nn.Linear(n_flatten, hidden_dim),
#        nn.ReLU(),
#        nn.Linear(hidden_dim, hidden_dim),
#        nn.ReLU(),
#        nn.Linear(hidden_dim, features_dim),
#        nn.ReLU()
#    )

  def forward(self, observations: th.Tensor, deterministic=False) -> th.Tensor:
    # 判断是否使用ε-greedy策略
#    if not deterministic and th.rand(1) < self.epsilon:
#      return th.randn(self.linear(self.cnn(observations)).shape)
    return self.linear(self.cnn(observations))

class KongmingChessEnv(gym.Env):
    def __init__(self, rand=False):
        self.board = np.zeros((7, 7, 1), dtype=np.uint8) # 0表示未翻牌,1表示焦点位置,2表示候选点,3表示翻牌.4表示不可点击区域,
        for x in chain(range(0,2), range(5, 7)):
          for y in chain(range(0, 2), range(5, 7)):
            self.board[x][y] = 4 # 不可点击区域.
        self.reset(rand)
        self.action_space = gym.spaces.Discrete(7 * 7)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(7, 7, 1), dtype=np.uint8) # TODO: 是否需要修改为离散空间?

    def getTotalReward(self):
        return self.totalReward

    def reset(self, rand=False):
        for x in range(0, 7):
          for y in range(0, 7):
            if (x < 2 or x >= 5) and (y < 2 or y >= 5):
              continue
            self.board[x][y] = 0
        self.focus = None
        self.candidates = None
        self.steps = 0
        self.done = False
        self.totalReward = 0

        if rand == False:
          self.board[3][3] = 3
          self.remainings = 32
        else:
          # 产生1到31个随机位置掀开，并确保游戏没结束.
          while True:
              for x in range(0, 7):
                  for y in range(0, 7):
                      if (x < 2 or x >= 5) and (y < 2 or y >= 5):
                          continue
                      self.board[x][y] = 0

              numOpen = random.randint(1, 31)
              i = numOpen
              while i > 0:
                  if numOpen == 1:
                      coord = 16
                  else:
                      coord = random.randint(0, 32)
                  if coord >= 0 and coord <= 5:
                      x = (coord % 3) + 2
                      y = (coord // 3)
                  elif coord >= 6 and coord <= 26:
                      y = ((coord - 6) // 7) + 2
                      x = ((coord - 6) % 7)
                  elif coord >= 27 and coord <= 32:
                      x = ((coord - 27) % 3) + 2
                      y = ((coord - 27) // 3) + 5
                  if self.board[x][y] == 0:
                      i -= 1
                      self.board[x][y] = 3
              self.remainings = 33 - numOpen
              if not self.check_over():
                  break
        return self.board

    def getFinalReward(self):
      return 0
      if self.remainings == 1 and self.board[3][3] == 0:
        return math.pow(10, 7)
      else:
        return math.pow(10, 7 -  self.remainings)

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


    def getRemainings(self):
        return self.remainings

    def render(self):
        for i in range(0, 7):
            s = ""
            for j in range(0, 7):
              s += str(self.board[i][j])
              s += ","
            print(s)

    def getFocus(self):
        return self.focus

    def avgDelta(self, delta):
        return delta
        formerTotal = self.totalReward - delta
        formerSteps = self.steps - 1
        if formerSteps == 0:
            return float(self.totalReward) / self.steps
        else:
            return float(self.totalReward) / self.steps - float(formerTotal) / formerSteps

    def step(self, action):
      if self.remainings <= 5:
          print("Nice Stepping:", self.remainings, self.totalReward, self.steps, self.totalReward / float(self.steps))
      self.steps += 1
      x = action % 7
      y = action // 7
      if self.board[x][y] == 4: # 不可点击区域,惩罚
        self.totalReward -= 100
        return self.board, self.avgDelta(-100), self.done, {}
     
      if self.focus == None:
        if self.board[x][y] == 3: # 当前无焦点，却点到空格子, 惩罚
          self.totalReward -= 100
          return self.board, self.avgDelta(-100), self.done, {}

        # 逐个判断候选选点
        candidates = []
        midpoints = []
        if y - 2 >= 0 and self.board[x][y -1] == 0 and self.board[x][y - 2] == 3:
          candidates.append((x, y - 2))
          midpoints.append((x, y - 1))
        if y + 2 < 7 and self.board[x][y + 1] == 0 and self.board[x][y + 2] == 3:
          candidates.append((x, y + 2))
          midpoints.append((x, y + 1))
        if x - 2 >= 0 and self.board[x - 1][y] == 0 and self.board[x - 2][y] == 3:
          candidates.append((x - 2, y))
          midpoints.append((x - 1, y))
        if x + 2 < 7 and self.board[x + 1][y] == 0 and self.board[x + 2][y] == 3:
          candidates.append((x + 2, y))
          midpoints.append((x + 1, y))

        if len(candidates) == 0: # 点了不当的焦点，惩罚
          self.totalReward -= 100
          return self.board, self.avgDelta(-100), self.done, {}

        # 如果只有一个候选节点，则直接跳.
        if len(candidates) == 1:
          self.board[x][y] = 3
          self.board[midpoints[0][0]][midpoints[0][1]] = 3
          self.board[candidates[0][0]][candidates[0][1]] = 0
          self.remainings -= 1
          self.done = self.check_over()
          self.totalReward += 0.5
          if self.done == True:
            self.totalReward += self.getFinalReward()
            return self.board, self.avgDelta(0.5 + self.getFinalReward()), self.done, {}
          return self.board, self.avgDelta(0.5), self.done, {}   

        self.focus = (x, y)
        self.candidates = candidates
        self.board[x][y] = 1
        for xx, yy in candidates:
          self.board[xx][yy] = 2
        return self.board, self.avgDelta(0), self.done, {}

      # 已经有跳转起点了.
      # 判断点击的是否为候选节点
      if self.board[x][y] != 2:
        self.board[self.focus[0]][self.focus[1]] = 0
        self.focus = None
        for xx, yy in self.candidates:
          self.board[xx][yy] = 3
        self.candidates = None
        self.totalReward -= 100
        return self.board, self.avgDelta(-100), self.done, {}  # 点了非候选节点，惩罚

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
        self.totalReward += 1
        self.totalReward += self.getFinalReward()
        return self.board, self.avgDelta(1 + self.getFinalReward()), True, {}
      self.totalReward += 1
      return self.board, self.avgDelta(1), self.done, {}

def make_env():
        return KongmingChessEnv()

def predict_proba(model, state):
  obs = model.policy.obs_to_tensor(state)[0]
  dis = model.policy.get_distribution(obs)
  probs = dis.distribution.probs
  probs_np = probs.detach().numpy()
  probs = []
  s = 0.0
  for i in range(0, 49):
      probs.append(math.exp(probs_np[0][i]))
      s += probs[i]
  s /= 100
  for i in range(0, 49):
      probs[i] /= s
  for i in range(0, 7):
      print('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(probs[7 * i], probs[7 * i + 1], probs[7 * i + 2], probs[7 * i + 3], probs[7 * i + 4], probs[7 * i + 5], probs[7 * i + 6]), sep='\t')
  action = np.argmax(probs_np)
  prob = np.max(probs_np)
  return action, prob, probs

def roulette_wheel_selection(a):
  # 计算数组中所有元素的和
  S = sum(a)
  # 计算每个元素的概率值
  p = [x/S for x in a]
  # 生成一个随机数
  r = random.random()
  # 依次累加每个元素的概率值，直到累加和大于随机数为止
  s = 0
  for i in range(len(p)):
    s += p[i]
    if s >= r:
      return i, p[i]
  # 如果所有元素的概率值之和小于随机数，返回最后一个元素
  return len(a) - 1, p[-1]

if __name__ == '__main__':
  env = make_vec_env(make_env, n_envs=6, vec_env_cls=DummyVecEnv, vec_env_kwargs={})
  #env = DummyVecEnv([lambda: env])

  policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=49),
  )
  #model = PPO('MlpPolicy', env, verbose=1)
  model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs, n_steps=1, batch_size=6144)
  #model = PPO.load("kongmingchess_model", env=env, verbose=1)
  np.random.seed(1001)
  model.learn(total_timesteps=1000000, log_interval=500)
  model.save("kongmingchess_model")

  input("模型训练结束，请按任意键开始走棋!")
  env = KongmingChessEnv()
  done = False
  obs = env.reset(False)
  action_set = set([])
  action_set_focus = []
  for x in range(0, 7):
    for y in range(0, 7):
      action_set_focus.append(set([]))
#  obs, reward, done, info = env.step(22)
#  print(reward)
#  print(obs)
  steps = 0
  while not env.check_over():
    preFocus = env.getFocus()
    # 获取动作预测
    action, prob, probs = predict_proba(model, obs)
    maxarg = True
    if preFocus == None:
        testSet = action_set
    else:
        testSet = action_set_focus[preFocus[1] * 7 + preFocus[0]]
    while action in testSet:
      #action, prob = model.predict(obs, deterministic=False)
      action, prob = roulette_wheel_selection(probs)
      maxarg = False

    print(action, prob, maxarg)
    obs, reward, done, info = env.step(action)
    if reward < 0:
        if preFocus == None:
          action_set.add(action)
        else:
          action_set_focus[preFocus[1] * 7 + preFocus[0]].add(action)
    elif reward > 0:
        action_set = set([])
        action_set_focus = []
        for x in range(0, 7):
            for y in range(0, 7):
                action_set_focus.append(set([]))
    steps += 1
    print('Action: ', action % 7, action // 7)
    print('Reward: ', reward)
    print('Remainings: ', env.getRemainings())
    print('Observation: ')
    env.render()
    print('Steps: ', steps)
    print('TotalReward: ', env.getTotalReward())
    lastAction = action
    time.sleep(0.1)
