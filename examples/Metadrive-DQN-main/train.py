# Import necessary libraries and modules
import numpy as np
import torch
import torch.nn as nn
from metadrive import MetaDriveEnv
from DuelDQN_Net import DuelDQN_Net
from torch.utils.tensorboard import SummaryWriter

# Hyper Parameters
BATCH_SIZE = 64
LR = 2e-4  # learning rate
EPSILON = 0.1  # greedy policy
SETTING_TIMES = 500  # greedy setting times
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 1000  # target update frequency
MEMORY_CAPACITY = 50000
MAX_EPISODES = 10000  # maximum episodes for training
MAX_STEPS = 1000  # maximum steps per episode

# GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the Duel DQN class
class DuelDQN(object):
    def __init__(self, is_train=True):
        self.IS_TRAIN = is_train
        self.eval_net, self.target_net = DuelDQN_Net().to(device), DuelDQN_Net().to(device)
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, self.eval_net.N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./log') if self.IS_TRAIN else None

        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.EPSILON = EPSILON if self.IS_TRAIN else 1.0
        self.SETTING_TIMES = SETTING_TIMES
        self.GAMMA = GAMMA
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.N_STATES = self.eval_net.N_STATES

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        if np.random.uniform() < self.EPSILON:  # greedy
            action_value = self.eval_net.forward(x).cpu()
            action_index = torch.max(action_value, 1)[1].data.numpy()[0]
            action_max_value = torch.max(action_value, 1)[0].data.numpy()[0]
        else:
            action_index = np.random.randint(0, self.eval_net.N_ACTIONS)
            action_max_value = 0
        return action_index, action_max_value

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.eval_net.N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, self.eval_net.N_STATES:self.eval_net.N_STATES + 1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, self.eval_net.N_STATES + 1:self.eval_net.N_STATES + 2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.eval_net.N_STATES:]).to(device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_eval_max_a = self.eval_net(b_s_).detach()
        eval_max_a_index = q_eval_max_a.max(1)[1].view(BATCH_SIZE, 1)

        q_next = self.target_net(b_s_).gather(1, eval_max_a_index)
        q_target = b_r + GAMMA * q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.IS_TRAIN and self.learn_step_counter % 100000 == 0:
            self.writer.add_scalar('Loss', loss.cpu(), self.learn_step_counter)

    def save(self, path):
        torch.save(self.eval_net.state_dict(), path)

    def load(self, path):
        self.eval_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))


# Main training loop
if __name__ == '__main__':
    config = dict(
        use_render=True,
        manual_control=False,
        traffic_density=0.1,
        num_scenarios=10000,
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=True, show_navi_mark=False, show_line_to_navi_mark=False),
        map="SSS",
    )
    env = MetaDriveEnv(config)
    dqn = DuelDQN(is_train=True)
    print('--------------\nCollecting experience...\n--------------')

    for i_episode in range(MAX_EPISODES):
        s = env.reset()
        total_reward = 0
        for t in range(MAX_STEPS):
            action_index, _ = dqn.choose_action(s)
            action = [action_index % 2 * 2 - 1, action_index // 2 * 2 - 1]
            s_, reward, done, _ = env.step(action)
            dqn.store_transition(s, action_index, reward, s_)
            total_reward += reward

            if dqn.memory_counter > dqn.MEMORY_CAPACITY:
                dqn.learn()

            if done:
                print(f"Episode {i_episode}, Reward: {total_reward}")
                break

            s = s_

        if i_episode % 100 == 0:
            dqn.save(f"model_{i_episode}.pth")

    env.close()


