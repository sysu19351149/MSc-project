import metadrive  # Import this package to register the environment!
import gymnasium as gym
from metadrive.envs.top_down_env import TopDownMetaDrive
gym.register(id="MetaDrive-topdown", entry_point=TopDownMetaDrive, kwargs=dict(config={}))

import numpy as np
import numpy.typing as npt

NUM_ACTIONS = 4

def discrete2continuous(action:int) -> npt.NDArray[np.float32]:
    """
    Convert discrete action to continuous action
    """
    assert 0 <= action < NUM_ACTIONS
    throttle_magnitude = 1.0
    brake_magnitude = 1.0
    steering_magnitude = 1.0
    if action == 0:
        return np.array([steering_magnitude, 0.0])
    elif action == 1:
        return np.array([0.0, throttle_magnitude])
    elif action == 2:
        return np.array([0.0, -brake_magnitude])
    elif action == 3:
        return np.array([-steering_magnitude, 0.0])
    else:
        raise ValueError("Invalid action: {}".format(action))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Q network
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # input is size 84x84x5
        # output is size NUM_ACTIONS
        self.conv1 = nn.Conv2d(5, 16, kernel_size=8, stride=4) # 84x84x5 -> 20x20x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) # 20x20x16 -> 9x9x32
        self.fc1 = nn.Linear(9*9*32, 256) # 9x9x32 -> 256
        self.fc2 = nn.Linear(256, NUM_ACTIONS) # 256 -> NUM_ACTIONS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x)) # Bx84x84x5 -> Bx20x20x16
        x = F.relu(self.conv2(x)) # Bx20x20x16 -> Bx9x9x32
        x = torch.flatten(x, start_dim=1) # Bx9x9x32 -> Bx9*9*32
        x = F.relu(self.fc1(x)) # Bx9*9*32 -> Bx256
        x = self.fc2(x) # Bx256 -> BxNUM_ACTIONS
        return x


def obs_batch_to_tensor(obs: List[npt.NDArray[np.float32]], device: torch.device) -> torch.Tensor:
    """
    Reshape the image observation from (B, H, W, C) to (B, C, H, W) and convert it to a tensor
    """
    return torch.tensor(np.stack(obs), dtype=torch.float32, device=device).permute(0, 3, 1, 2)


def deviceof(m: nn.Module) -> torch.device:
    """
    Get the device of the given module
    """
    return next(m.parameters()).device


class QPolicy:
    def __init__(self, net: QNetwork, epsilon: float = 0.0):
        self.net = net
        self.epsilon = epsilon

    def __call__(self, obs: npt.NDArray[np.float32]) -> int:
        """
        Return action given observation
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        else:
            device = deviceof(self.net)
            obs_batch = obs_batch_to_tensor([obs], device)
            with torch.no_grad():
                q_values = self.net(obs_batch)[0]
                return torch.argmax(q_values).item()


import typing


class Transition(typing.NamedTuple):
    obs: npt.NDArray[np.float32]
    action: int
    reward: float
    next_obs: npt.NDArray[np.float32]
    terminated: bool


def collect_trajectory(env: gym.Env, policy: typing.Callable[[npt.NDArray], int]) -> List[Transition]:
    """
    Collect a trajectory from the environment using the given policy
    """
    trajectory = []

    obs, info = env.reset()

    while True:
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(discrete2continuous(action))
        trajectory.append(Transition(obs, action, reward, next_obs, terminated))
        if terminated or truncated:
            break
        else:
            obs = next_obs

    return trajectory


import random


class ReplayBuffer():
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0

    def push(self, transition: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        # Overwrite the earliest stuff if the buffer is full.
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def compute_target_q_values(
        primary_q_network: QNetwork,
        target_q_network: QNetwork,
        rewards: List[float],
        next_observations: List[npt.NDArray[np.float32]],
        terminateds: List[bool],
        gamma: float
) -> List[float]:
    """
    Compute the target Q values for the transitions
    """
    # assert that the lengths of the lists are the same
    n_transitions = len(rewards)
    assert len(next_observations) == n_transitions
    assert len(terminateds) == n_transitions

    # assert that primary and target networks are on the same device
    device = deviceof(primary_q_network)
    assert deviceof(target_q_network) == device

    # compute the Q_target(s', a') values
    next_obs_batch = obs_batch_to_tensor(next_observations, device)
    with torch.no_grad():
        primary_next_q_values = primary_q_network(next_obs_batch)
        target_next_q_values = target_q_network(next_obs_batch)

    # compute argmax_a' Q_phi(s', a') for each transition
    argmax_next_q_values = torch.max(primary_next_q_values, dim=1).indices

    # compute the Q_phi_target(s', argmax_a' Q_phi(s', a')) values
    max_next_q_values = target_next_q_values[range(n_transitions), argmax_next_q_values]

    # compute the target Q values
    target_q_values = []
    for reward, terminated, max_next_q_value in zip(rewards, terminateds, max_next_q_values):
        if terminated:
            target_q_values.append(reward)
        else:
            target_q_values.append(reward + gamma * float(max_next_q_value))

    return target_q_values


def train_primary_q_network(
        primary_q_network: QNetwork,
        target_q_network: QNetwork,
        optimizer: torch.optim.Optimizer,
        transitions: List[Transition],
        gamma: float
) -> float:
    """
    Train the Q network using the given transitions
    """
    # unpack the transitions
    obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch = zip(*transitions)

    device = deviceof(primary_q_network)

    # compute the target Q values
    target_q_values = compute_target_q_values(primary_q_network, target_q_network, reward_batch, next_obs_batch,
                                              terminated_batch, gamma)
    target_q_values_tensor = torch.tensor(target_q_values, dtype=torch.float32, device=device)

    # compute the Q values for the picked actions
    obs_batch = obs_batch_to_tensor(obs_batch, device)
    all_q_values = primary_q_network(obs_batch)
    q_values = all_q_values[range(len(action_batch)), action_batch]

    # compute the loss
    loss = F.mse_loss(q_values, target_q_values_tensor)

    # perform gradient descent
    optimizer.zero_grad()# scatter plot of the returns


def soft_update_target_q_network(primary_q_network: QNetwork, target_q_network: QNetwork, tau: float):
    """
    Soft update the target Q network
    """
    # assert that primary and target networks are on the same device
    device = deviceof(primary_q_network)
    assert deviceof(target_q_network) == device

    # update the target network parameters
    for primary_param, target_param in zip(primary_q_network.parameters(), target_q_network.parameters()):
        target_param.data.copy_(tau * primary_param.data + (1 - tau) * target_param.data)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Q network for a dueling dqn
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # input is size 84x84x5
        # output is size NUM_ACTIONS
        self.conv1 = nn.Conv2d(5, 16, kernel_size=8, stride=4) # 84x84x5 -> 20x20x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) # 20x20x16 -> 9x9x32
        self.fc1 = nn.Linear(9*9*32, 256) # 9x9x32 -> 256
        self.advantage_head = nn.Linear(256, NUM_ACTIONS) # 256 -> NUM_ACTIONS
        self.value_head = nn.Linear(256, 1) # 256 -> 1

def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = F.relu(self.conv1(x)) # Bx84x84x5 -> Bx20x20x16
    x = F.relu(self.conv2(x)) # Bx20x20x16 -> Bx9x9x32
    x = torch.flatten(x, start_dim=1) # Bx9x9x32 -> Bx9*9*32
    x = F.relu(self.fc1(x)) # Bx9*9*32 -> Bx256
    advantage = self.advantage_head(x) # Bx256 -> BxNUM_ACTIONS
    value = self.value_head(x) # Bx256 -> Bx1
    return value + advantage - advantage.mean(dim=1, keepdim=True)
QNetwork.forward = forward

# disable logging from metadrive
import logging
import inspect
import metadrive.envs.top_down_env
logging.getLogger(inspect.getfile(metadrive.envs.base_env)).setLevel(logging.WARNING)

def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

primary_q_network = QNetwork().to(device)
target_q_network = QNetwork().to(device)

q_optimizer = torch.optim.Adam(primary_q_network.parameters())

policy = QPolicy(primary_q_network, epsilon=0.3)

replay_buffer = ReplayBuffer()

step = 0
returns = []
losses = []
env = gym.make("MetaDrive-topdown", config={"use_render": False, "horizon": 300, "num_scenarios": 100})

TRAIN_EPOCHS = 1000
EPISODES_PER_BATCH = 16
UPDATE_TO_COLLECT_RATIO = 1.0
GAMMA = 0.80
TAU = 0.01
set_lr(q_optimizer, 1e-4)

# Train
while step < TRAIN_EPOCHS:
    trajectory_returns = []

    for _ in range(EPISODES_PER_BATCH):
        # Collect trajectory
        transitions = collect_trajectory(env, policy)
        rew_traj = [t.reward for t in transitions]

        # Update replay buffer
        for t in transitions:
            replay_buffer.push(t)

        # Update trajectory returns
        trajectory_returns.append(sum(rew_traj))

    loss = train_primary_q_network(
        primary_q_network,
        target_q_network,
        q_optimizer,
        replay_buffer.sample(round(EPISODES_PER_BATCH * UPDATE_TO_COLLECT_RATIO)),
        GAMMA
    )
    soft_update_target_q_network(primary_q_network, target_q_network, TAU)

    # slowly decay the epsilon
    policy.epsilon = policy.epsilon * 0.998

    # collect statistics
    returns.append(trajectory_returns)
    losses.append(loss)

    # print(f"Step {step}, Avg. Returns: {np.mean(trajectory_returns):.3f} +/- {np.std(trajectory_returns):.3f}, Median: {np.median(trajectory_returns):.3f}, Q-Network Loss: {losses[-1]:.3f}")

    step += 1

env.close()

env = gym.make("MetaDrive-topdown", config={"use_render": True, "horizon": 500, "num_scenarios": 100})
transitions = collect_trajectory(env, QPolicy(primary_q_network, epsilon=0.0))
env.close()

rew = [t.reward for t in transitions]

print("Reward:", sum(rew))
env.close()

import matplotlib.pyplot as plt

def moving_average(x: List[float], window_size: int) -> List[float]:
    """
    Compute the moving average of the given list
    """
    return [np.mean(x[i:i+window_size]) for i in range(len(x) - window_size + 1)]

return_medians = [np.median(returns[i]) for i in range(len(returns))]
return_means = [np.mean(returns[i]) for i in range(len(returns))]
return_stds = [np.std(returns[i]) for i in range(len(returns))]
plt.plot(return_medians, label="Median")
plt.plot(return_means, label="Mean")
plt.fill_between(range(len(return_means)), np.array(return_means) - np.array(return_stds), np.array(return_means) + np.array(return_stds), alpha=0.3)
plt.xlabel("Epoch")
plt.ylabel("Average Return")
plt.legend()
plt.show()

# scatter plot of the returns
xs = []
ys = []
for t, rets in enumerate(returns):
    for ret in rets:
        xs.append(t)
        ys.append(ret)
plt.scatter(xs, ys, alpha=0.2)