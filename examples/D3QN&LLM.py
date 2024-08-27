import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
from metadrive import MetaDriveEnv
import argparse
import logging
import random
from pynput.keyboard import Key, Controller
import base64
import requests

import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE

def encode_image(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# 设置OpenAI的API密钥
api_key = '****'


# Function to send image to OpenAI and get the response
def get_reward_from_gpt(image, str1):
    base64_image = encode_image(image)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = """             
    I'm running the metadrive simulation script, assuming you are an autodrive agent.
Next I'll give you the camera image information, the camera is in front of the car, as well as velocity information, 
steering information, acceleration information, and the current reward value.
I need you to evaluate the current situation based on these inputs and provide feedback on the appropriateness of the reward.
You should consider the vehicle's speed, the direction it's moving, and the overall safety and efficiency of the drive.
Your output should be a numerical value indicating the adjusted reward (if necessary) and  you do not need to explanation for your decision.
    """
    prompt += str1

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def get_reward_from_gpt(image, str1):
    base64_image = encode_image(image)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = """             
    I'm running the metadrive simulation script, assuming you are an autodrive agent.
Next I'll give you the camera image information, the camera is in front of the car, as well as velocity information, 
steering information, acceleration information, and the current reward value.
I need you to evaluate the current situation based on these inputs and provide feedback on the appropriateness of the reward.
You should consider the vehicle's speed, the direction it's moving, and the overall safety and efficiency of the drive.
Your output should be a numerical value between -1 to 1 indicating the  reward and  you do not need to explanation for your decision.
    """
    prompt += str1

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


# 定义Q网络
def create_q_network(state_size, action_size):
    model = tf.keras.Sequential([
        layers.Dense(64, input_dim=state_size, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation=None)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model


# 定义D3QN Agent
class D3QNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, batch_size=64, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, tau=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.batch_size = batch_size

        self.memory = deque(maxlen=2000)

        # 初始化Q网络和目标Q网络
        self.qnetwork_local = create_q_network(state_size, action_size)
        self.qnetwork_target = create_q_network(state_size, action_size)

        # 将目标网络的权重初始化为与局部网络相同
        self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())

        # 生成离散动作集
        self.action_space = self.create_discrete_action_space()

    def create_discrete_action_space(self):
        steering_actions = [-1.0, -0.5, 0.0, 0.5, 1.0]  # Steering actions
        throttle_actions = [-1.0, 0.0, 1.0]  # Throttle actions
        return np.array(np.meshgrid(steering_actions, throttle_actions)).T.reshape(-1, 2)

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            self.learn()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.qnetwork_local.predict(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])

    def learn(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        actions = np.array(actions)
        dones = np.array(dones).astype(int)

        # 获取下一个状态的最大Q值 (从目标网络)
        q_values_next = self.qnetwork_target.predict(next_states)
        max_q_values_next = np.max(q_values_next, axis=1)

        # 计算目标Q值
        target_q_values = rewards + (1 - dones) * self.gamma * max_q_values_next

        # 获取当前状态的Q值
        q_values = self.qnetwork_local.predict(states)

        # 更新动作对应的Q值
        for i in range(self.batch_size):
            q_values[i][actions[i]] = target_q_values[i]

        # 训练Q网络
        self.qnetwork_local.fit(states, q_values, epochs=1, verbose=0)

        # 更新目标网络
        self.update_target_network()

        # 减少epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        local_weights = self.qnetwork_local.get_weights()
        target_weights = self.qnetwork_target.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * local_weights[i] + (1 - self.tau) * target_weights[i]
        self.qnetwork_target.set_weights(target_weights)

    def choose_action(self, action_idx):
        return self.action_space[action_idx]

    def save_model(self, filepath='d3qn_model.h5'):
        self.qnetwork_local.save_weights(filepath)
        print(f"Model weights saved to {filepath}")

    def load_model(self, filepath='d3qn_model.h5'):
        self.qnetwork_local.load_weights(filepath)
        self.qnetwork_target.load_weights(filepath)
        print(f"Model weights loaded from {filepath}")


# 训练D3QN Agent
def train_d3qn(env, agent, n_episodes=1000, max_t=1000):
    scores = []
    for e in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action_idx = agent.act(state)
            action = agent.choose_action(action_idx)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action_idx, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        print(f"Episode {e}/{n_episodes}, Score: {score}")

        # 每隔100轮保存一次模型
        if e % 100 == 0:
            agent.save_model(filepath=f'd3qn_model_mid.h5')
    return scores


# 主程序
if __name__ == "__main__":
    env = MetaDriveEnv(dict(use_render=False))  # 初始化MetaDrive环境
    state_size = env.observation_space.shape[0]  # 获取状态空间大小
    action_size = 15  # 5 (steering actions) * 3 (throttle actions)
    agent = D3QNAgent(state_size, action_size)  # 初始化D3QN代理

    # 如果有保存的模型，可以选择加载
    # agent.load_model(filepath='d3qn_model_100.h5')

    # 训练D3QN代理
    scores = train_d3qn(env, agent)

    # 测试训练后的Agent
    state, _ = env.reset()
    done = False
    while not done:
        action_idx = agent.act(state)
        action = agent.choose_action(action_idx)
        next_state, reward, done, _, _ = env.step(action)
        env.render()
        state = next_state

    env.close()  # 关闭环境

    # 保存最终的模型
    agent.save_model(filepath='d3qn_model_final.h5')

import numpy as np
import gym
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import os
from PIL import Image

import metadrive
from metadrive.envs import MetaDriveEnv



def gpt_evaluate(image, text_description):

    return random.uniform(0, 1)


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

            # 提取图像状态和生成文本描述
            image = env.get_image()  # 假设环境支持图像提取
            text_description = f"Current speed: {env.vehicle.get_speed()}"

            # 使用GPT-4进行评估
            gpt_reward = gpt_evaluate(image, text_description)
            reward += gpt_reward  # 将GPT的评价加入奖励

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