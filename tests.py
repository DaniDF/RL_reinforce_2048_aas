import os.path
import time
import re

import numpy as np
import matplotlib.pyplot as plt

import utils
from game import Game
from reinforce_base_agent import ReinforceBaseAgent
from data_logger import DataLogger
from replay_games import ReplayGames


def play_episode(agent):
    game = Game()
    observation, _ = game.reset()

    terminated = False

    while not terminated:
        observation = observation.flatten() / game.MAX_TILE
        if agent is not None:
            action = agent.predict(observation)
        else:
            action = game.get_random_action()
        observation, reward, terminated, _, _ = game.step(action)

    return game


def play_games(n_episodes=1000, ckpt_path=".", ckpt_filename=None, save_rewards=False, save_ending_states=False):
    agent = ReinforceBaseAgent(input_shape=(4 * 4,), output_space=range(4))
    # agent = DQNAgent(input_shape=(4*4,), output_space=range(4))

    if ckpt_filename is None:
        model_weight_file = utils.get_last_ckpt_name(ckpt_path=ckpt_path)
    else:
        model_weight_file = os.path.join(ckpt_path, ckpt_filename)

    agent.load_model_weights(model_weight_file)

    print("Using ckpt file: {}".format(model_weight_file))

    rewards = []
    ending_states = []
    best_score = 0
    tiles = {}

    for count_episode in range(n_episodes):
        game = play_episode(agent)
        rewards.append(game.game_score)
        utils.update_perc(count_episode, n_episodes)

        if game.game_score > best_score:
            best_score = game.game_score
        max_tile = game.get_max_tile()
        if max_tile not in tiles:
            tiles[max_tile] = 0
        tiles[max_tile] += 1

        if save_ending_states:
            ending_states.append((game.observation_space.tolist(), game.game_score.item()))

    plt.plot(rewards, ".", color="red")
    plt.show()

    tiles_keys = list(tiles.keys())
    tiles_keys.sort()
    tiles = {i: tiles[i] for i in tiles_keys}

    plt.bar(list(tiles.keys()), tiles.values(), color="red")
    plt.show()

    print("\nBest score:", best_score)
    print("Percentile 95:", np.percentile(rewards, 5))

    if save_rewards:
        utils.save_data_in_file([i.item() for i in rewards], os.path.join(ckpt_path, f"play_games_{n_episodes}_rewards.json"))

    if save_ending_states:
        utils.save_data_in_file(ending_states, os.path.join(ckpt_path, f"play_games_{n_episodes}_ending_states.json"))


def play_game_show(ckpt_path=".", ckpt_filename=None, replay_games_file=None):
    agent = ReinforceBaseAgent(input_shape=(4 * 4,), output_space=range(4))

    if ckpt_filename is None:
        model_weight_file = utils.get_last_ckpt_name(ckpt_path=ckpt_path)
    else:
        model_weight_file = os.path.join(ckpt_path, ckpt_filename)

    agent.load_model_weights(model_weight_file)

    print("Using ckpt file: {}".format(model_weight_file))

    action_translator = ["LEFT", "UP", "RIGHT", "DOWN"]

    game = Game()
    if replay_games_file is not None:
        replay = ReplayGames(os.path.join(ckpt_path, replay_games_file))
        observation = replay.sample_game()
        game.observation_space = observation
    else:
        observation, _ = game.reset()

    terminated = False

    while not terminated:
        print(game)
        observation = observation.flatten() / game.MAX_TILE
        action = agent.predict(observation)
        observation, reward, terminated, _, _ = game.step(action)

        print(action_translator[action], game.game_score)

        time.sleep(2)


def load_data_log(data_log_path=".", flag_plot_results=True):
    data_logger = DataLogger(1)

    data_logger.losses_policy = utils.load_data_from_file(os.path.join(data_log_path, "losses_policy.json"))
    data_logger.losses_value = utils.load_data_from_file(os.path.join(data_log_path, "losses_value.json"))
    data_logger.deltas_policy = utils.load_data_from_file(os.path.join(data_log_path, "deltas_policy.json"))
    data_logger.rewards = utils.load_data_from_file(os.path.join(data_log_path, "rewards.json"))
    data_logger.d_reward = utils.load_data_from_file(os.path.join(data_log_path, "d_reward.json"))
    data_logger.state_values = utils.load_data_from_file(os.path.join(data_log_path, "state_values.json"))
    data_logger.log_prob = utils.load_data_from_file(os.path.join(data_log_path, "log_prob.json"))

    if flag_plot_results:
        utils.plot_results(data_logger.losses_policy, data_logger.losses_value, data_logger.deltas_policy,
                           data_logger.d_reward, data_logger.state_values, data_logger.log_prob, data_logger.rewards)


def compute_percentiles(n_episodes=1000, ckpt_path=".", flag_plot_results=True, save_percentile=False):
    filenames = [file for file in os.listdir(ckpt_path) if os.path.isfile(os.path.join(ckpt_path, file))]

    ckpt_filenames = set()

    for file in filenames:
        match = re.search("cp-\\d{3}.ckpt", file)
        if match is not None:
            ckpt_filenames.add(os.path.join(ckpt_path, match.group()))

    ckpt_filenames = sorted(list(ckpt_filenames))

    agent = ReinforceBaseAgent(input_shape=(4 * 4,), output_space=range(4))
    rewards_percentile = []
    for ckpt in ckpt_filenames:
        print("\nUsing file:", ckpt)
        agent.load_model_weights(ckpt)

        rewards = []
        for count_episode in range(n_episodes):
            game = play_episode(agent)
            rewards.append(game.game_score)
            utils.update_perc(count_episode, n_episodes)

        rewards_percentile.append(np.percentile(rewards, 5))

    if flag_plot_results:
        utils.plot_data(rewards_percentile, "rewards_percentile", "purple")

    if save_percentile:
        utils.save_data_in_file(rewards_percentile,
                                os.path.join(ckpt_path, f"compute_percentiles_{n_episodes}_rewards.json"))


def plot_file_json(filename, color="purple"):
    data = utils.load_data_from_file(filename)
    utils.plot_data(data, os.path.basename(filename), color)


if __name__ == "__main__":
    play_games(n_episodes=int(1e3), ckpt_path="ckpt-#0-2_norm_big-net", ckpt_filename="cp-010.ckpt", save_rewards=True, save_ending_states=True)
    # play_game_show(ckpt_path="ckpt-#1-1_norm_huge-net_replay_games", ckpt_filename=None, replay_games_file=None)
    # load_data_log(data_log_path="ckpt-#0-2_norm_big-net", flag_plot_results=True)
    # compute_percentiles(n_episodes=int(1e3), ckpt_path="ckpt-#0-2_norm_big-net", flag_plot_results=False, save_percentile=True)
    # plot_file_json(os.path.join("ckpt-#0-2_norm_big-net", "compute_percentiles_1000_rewards.json"))
