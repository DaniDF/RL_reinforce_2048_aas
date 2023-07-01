from threading import Thread

from game import Game


def play_episode(agent, render=None):
    cumulative_reward = 0
    replay = []

    game = Game()
    observation, _ = game.reset()

    terminated = False

    while not terminated:
        if render is not None:
            print(game)

        observation = observation.flatten() / game.MAX_TILE
        action = agent.predict(observation)
        prev_observation = observation
        observation, reward, terminated, _, _ = game.step(action)
        cumulative_reward += reward

        replay.append([prev_observation, observation.flatten() / game.MAX_TILE, reward, action, True])

    for experience in replay:
        reward = experience[2]
        experience[2] = cumulative_reward
        cumulative_reward -= reward

    return replay[0][2], replay


class PolicyThread:
    def __init__(self, agent, data_logger, num_episodes):
        self.__agent__ = agent
        self.__data_logger__ = data_logger
        self.__num_episodes__ = num_episodes

        self.worker = None

    def start_worker_thread(self, alpha_policy=1.0, alpha_value=1.0):
        self.worker = Thread(target=self.__do_job__, args=(alpha_policy, alpha_value))
        self.worker.start()

    def join_worker_thread(self):
        self.worker.join()

    def __do_job__(self, alpha_policy=1.0, alpha_value=1.0):
        for count_episode in range(self.__num_episodes__):
            reward, episode = play_episode(self.__agent__)

            loss_policy, loss_value, deltas_policy, d_reward, state_values, log_prob = self.__agent__.train(episode,
                                                                                                            alpha_policy=alpha_policy,
                                                                                                            alpha_value=alpha_value)

            self.__data_logger__.append_data(loss_policy, loss_value, deltas_policy, reward, d_reward, state_values,
                                             log_prob)
            self.__data_logger__.notify_end_episode()