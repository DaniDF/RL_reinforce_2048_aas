import os.path
import time
import json
import re

import numpy as np
import matplotlib.pyplot as plt

import utils
from game import Game
from reinforce_base_agent import ReinforceBaseAgent
from data_logger import DataLogger


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


def play_games(n_episodes=1000, ckpt_path=".", ckpt_filename=None, save_in_file=False):
    agent = ReinforceBaseAgent(input_shape=(4 * 4,), output_space=range(4))
    # agent = DQNAgent(input_shape=(4*4,), output_space=range(4))

    if ckpt_filename is None:
        model_weight_file = utils.get_last_ckpt_name(ckpt_path=ckpt_path)
    else:
        model_weight_file = os.path.join(ckpt_path, ckpt_filename)

    agent.load_model_weights(model_weight_file)

    print("Using ckpt file: {}".format(model_weight_file))

    rewards = []
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

    rewards_random = []
    best_score_random = 0
    tiles_random = {}

    for count_episode in range(n_episodes):
        game = play_episode(None)

        rewards_random.append(game.game_score)
        utils.update_perc(count_episode, n_episodes)

        if game.game_score > best_score_random:
            best_score_random = game.game_score
        max_tile = game.get_max_tile()
        if max_tile not in tiles_random:
            tiles_random[max_tile] = 0
        tiles_random[max_tile] += 1

    plt.plot(rewards, ".", color="red")
    plt.plot(rewards_random, ".", color="blue")
    plt.show()

    tiles_keys = list(tiles.keys())
    tiles_keys.sort()
    tiles = {i: tiles[i] for i in tiles_keys}

    tiles_keys_random = list(tiles_random.keys())
    tiles_keys_random.sort()
    tiles_random = {i: tiles_random[i] for i in tiles_keys_random}

    plt.bar(list(tiles.keys()), tiles.values(), color="red")
    plt.show()

    print("\nBest score:", best_score)
    print("Percentile 95:", np.percentile(rewards, 5))
    print("Random percentile 95:", np.percentile(rewards_random, 5))

    if save_in_file:
        with open(f"play_games_{n_episodes}_rewards.json") as file_out:
            json.dump(rewards, file_out)


def play_game_show(ckpt_path=".", ckpt_filename=None):
    agent = ReinforceBaseAgent(input_shape=(4 * 4,), output_space=range(4))

    if ckpt_filename is None:
        model_weight_file = utils.get_last_ckpt_name(ckpt_path=ckpt_path)
    else:
        model_weight_file = os.path.join(ckpt_path, ckpt_filename)

    agent.load_model_weights(model_weight_file)

    print("Using ckpt file: {}".format(model_weight_file))

    action_translator = ["LEFT", "UP", "RIGHT", "DOWN"]

    game = Game()
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

    with open(os.path.join(data_log_path, "losses_policy.json"), "r") as file_in:
        data_logger.losses_policy = json.load(file_in)

    with open(os.path.join(data_log_path, "losses_value.json"), "r") as file_in:
        data_logger.losses_value = json.load(file_in)

    with open(os.path.join(data_log_path, "deltas_policy.json"), "r") as file_in:
        data_logger.deltas_policy = json.load(file_in)

    with open(os.path.join(data_log_path, "rewards.json"), "r") as file_in:
        data_logger.rewards = json.load(file_in)

    with open(os.path.join(data_log_path, "d_reward.json"), "r") as file_in:
        data_logger.d_reward = json.load(file_in)

    with open(os.path.join(data_log_path, "state_values.json"), "r") as file_in:
        data_logger.state_values = json.load(file_in)

    with open(os.path.join(data_log_path, "log_prob.json"), "r") as file_in:
        data_logger.log_prob = json.load(file_in)

    if flag_plot_results:
        utils.plot_results(data_logger.losses_policy, data_logger.losses_value, data_logger.deltas_policy,
                           data_logger.d_reward, data_logger.state_values, data_logger.log_prob, data_logger.rewards)


def compute_percentiles(n_episodes=1000, ckpt_path=".", flag_plot_results=True, save_in_file=False):
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
        plt.plot(rewards_percentile, ".", color="purple")
        plt.title("rewards_percentile")
        plt.show()

    if save_in_file:
        with open(f"compute_percentiles_{n_episodes}_rewards.json") as file_out:
            json.dump(rewards_percentile, file_out)


if __name__ == "__main__":
    # play_games(n_episodes=int(1e3), ckpt_path="ckpt-#0-3_norm_huge-net", ckpt_filename="cp-004.ckpt")
    # play_game_show(ckpt_path="ckpt-#0-3_norm_huge-net", ckpt_filename="cp-004.ckpt")
    # load_data_log(data_log_path="ckpt-#3-0_small-training-small-net", flag_plot_results=True)
    compute_percentiles(n_episodes=int(1e3), ckpt_path="ckpt-#0-3_norm_huge-net",
                        flag_plot_results=False, save_in_file=True)
