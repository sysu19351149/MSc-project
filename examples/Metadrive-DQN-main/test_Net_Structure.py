# Metadrive
from metadrive import MetaDriveEnv

# Other Libs
import numpy as np
import random
import matplotlib.pyplot as plt
from DQN import *
from DoubleDQN import *
from DuelDQN import *
from math import floor

# Init Metadrive Env
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
    map="SSS",  # 使用最新版本的 MetaDrive 中可用的地图
)


def choose_steering(action_index):
    steering_index = floor(action_index / 3)
    switch = {0: -0.5,
              1: 0.0,
              2: 0.5, }
    steering = switch.get(steering_index)
    return steering


def choose_acceleration(action_index):
    acceleration_index = floor(action_index % 3)
    switch = {0: -0.5,
              1: 0.0,
              2: 0.5, }
    acceleration = switch.get(acceleration_index)
    return acceleration


if __name__ == '__main__':
    env = MetaDriveEnv(config)
    dqn = DuelDQN(is_train=True)
    print('--------------\nCollecting experience...\n--------------')
    best_reward = 0

    rewards = []
    avg_rewards = []
    avg_q_values = []

    for i_episode in range(200000):
        if i_episode <= dqn.SETTING_TIMES:
            dqn.EPSILON = 0.1 + i_episode / dqn.SETTING_TIMES * (0.9 - 0.1)
        s = env.reset()
        s = s[: dqn.N_STATES]
        env.vehicle.expert_takeover = True
        # indirect params
        total_reward = 0
        total_action_value = 0
        action_counter = 0
        reward_counter = 0
        while True:
            # take action based on the current state
            action_index, action_value = dqn.choose_action(s)

            total_action_value += action_value
            if action_value != 0:
                action_counter += 1

            # choose action
            steering = choose_steering(action_index)
            acceleration = choose_acceleration(action_index)
            action = np.array([steering, acceleration])
            # step
            s_, reward, done, info = env.step(action)
            # slice s and s_
            s = s[: dqn.N_STATES]
            s_ = s_[: dqn.N_STATES]
            # store the transitions of states
            dqn.store_transition(s, action_index, reward, s_)

            total_reward += reward
            reward_counter += 1

            # if the experience repaly buffer is filled,
            # DQN begins to learn or update its parameters.
            if dqn.memory_counter > dqn.MEMORY_CAPACITY:
                dqn.learn()
            if done:
                # if game is over, then skip the while loop.
                if best_reward <= total_reward:
                    best_reward = total_reward
                    dqn.save("./" + str(round(best_reward)) + "check_points.tar")
                if i_episode % 1000 == 999:
                    dqn.save("./" + str(i_episode) + ".tar")
                print('Ep: ', i_episode, ' |', 'Ep_r: ', round(total_reward, 2), ' |', 'Best_r: ',
                      round(best_reward, 2))
                break
            else:
                # use next state to update the current state.
                s = s_

        # Logging for plotting
        rewards.append(total_reward)
        avg_rewards.append(total_reward / reward_counter)
        avg_q_values.append(total_action_value / action_counter if action_counter > 0 else 0)

        dqn.writer.add_scalar('Ep_r', total_reward, i_episode)
        dqn.writer.add_scalar('Ave_r', total_reward / reward_counter, i_episode)
        dqn.writer.add_scalar('Ave_Q_value', total_action_value / action_counter, i_episode)

    # Plotting reward and Q-value over episodes
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Total Reward")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over Episodes')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(avg_q_values, label="Average Q-value")
    plt.xlabel('Episode')
    plt.ylabel('Average Q-value')
    plt.title('Q-value over Episodes')
    plt.legend()

    plt.tight_layout()
    plt.show()