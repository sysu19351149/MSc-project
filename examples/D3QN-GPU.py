import numpy as np
import gym
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import os

import metadrive
from metadrive.envs import MetaDriveEnv
from metadrive.engine.engine_utils import initialize_engine, close_engine

# 安装 Prioritized Experience Replay
from keras_rl.memory import PrioritizedMemory

class D3QNAgent:
    def __init__(self, state_size, action_size, prioritized_replay=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedMemory(limit=2000, alpha=0.6) if prioritized_replay else deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.prioritized_replay = prioritized_replay

        # 定义模型和目标模型
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        if self.prioritized_replay:
            q_values = self.model.predict(state)[0]
            next_q_values = self.target_model.predict(next_state)[0]
            target = reward
            if not done:
                target += self.gamma * np.amax(next_q_values)
            td_error = abs(q_values[action] - target)
            self.memory.append((state, action, reward, next_state, done), td_error)
        else:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        if self.prioritized_replay:
            minibatch, idxs, is_weights = self.memory.sample(batch_size)
        else:
            minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.prioritized_replay:
                # 更新优先级
                self.memory.update(idxs[i], abs(target_f[0][action] - target))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def train_d3qn(episodes=1000, batch_size=32, save_interval=100):
    env = MetaDriveEnv(
        dict(
            use_render=False,
            environment_num=100,
            start_seed=0,
        )
    )

    state_size = env.observation_space.shape[0]
    steering_actions = [-0.5, 0.0, 0.5]
    throttle_actions = [-0.5, 0.0, 0.5]
    action_size = len(steering_actions) * len(throttle_actions)
    agent = D3QNAgent(state_size, action_size, prioritized_replay=True)

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action_index = agent.act(state)
            steer = steering_actions[action_index // 3]
            throttle = throttle_actions[action_index % 3]
            next_state, reward, done, _ = env.step([steer, throttle])
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action_index, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % save_interval == 0:
            agent.save(f"d3qn_model_{e}.h5")

    env.close()

# 开始训练
train_d3qn(episodes=1000, batch_size=32, save_interval=100)


