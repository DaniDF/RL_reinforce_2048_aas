from threading import Lock

from utils import update_perc


class DataLogger:
    def __init__(self, total_episodes):
        self.losses_policy = []
        self.losses_value = []
        self.deltas_policy = []
        self.rewards = []
        self.d_reward = []
        self.state_values = []
        self.log_prob = []

        self.__total_episodes__ = total_episodes
        self.__terminated_episodes__ = 0

        self.__data_lock__ = Lock()
        self.__notification_lock__ = Lock()

    def append_data(self, loss_policy, loss_value, delta_policy, reward, d_reward, state_values, log_prob):
        with self.__data_lock__:
            self.losses_policy.append(loss_policy)
            self.losses_value.append(loss_value)
            self.deltas_policy.append(delta_policy)
            self.rewards.append(reward)
            self.d_reward.append(d_reward)
            self.state_values.append(state_values)
            self.log_prob.append(log_prob)

    def notify_end_episode(self):
        with self.__notification_lock__:
            self.__terminated_episodes__ += 1

            update_perc(self.__terminated_episodes__, self.__total_episodes__)

    def reset_episode_counter(self):
        with self.__notification_lock__:
            self.__terminated_episodes__ = 0

            update_perc(self.__terminated_episodes__, self.__total_episodes__)
