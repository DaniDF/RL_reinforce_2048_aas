import utils
import os

import numpy as np
import matplotlib.pyplot as plt


def filter_ending_states(filename, threshold):
    data = utils.load_data_from_file(filename)

    data_filtered = [d for d in data if np.array(d[0]).flatten().max() >= threshold]

    utils.save_data_in_file(data_filtered,
                            os.path.join(os.path.dirname(filename), f"min_{threshold}_{os.path.basename(filename)}"))


def plot_max_tile(filename):
    data = utils.load_data_from_file(filename)

    tiles_counter = {}

    for d in data:
        max_tile = np.array(d[0]).max()
        if max_tile not in tiles_counter:
            tiles_counter[max_tile] = 0
        tiles_counter[max_tile] += 1

    plt.bar(list(tiles_counter.keys()), tiles_counter.values(), color="red")
    plt.show()


def plot_and_save(ckpt_dir="ckpt-#0-1_big-net", filename="d_reward.json"):
    data = utils.load_data_from_file(os.path.join(ckpt_dir, filename))

    plt.plot(data, ".", color="orange")
    plt.xlabel("# episodes")
    plt.savefig(os.path.join(ckpt_dir, f'{filename.split(".")[0]}.png'))
    plt.show()


def plot_bar_and_save(ckpt_dir="ckpt-#0-1_big-net", filename="d_reward.json"):
    data = utils.load_data_from_file(os.path.join(ckpt_dir, filename))

    data_filtered = [np.array(d[0]).flatten().max() for d in data]

    tiles = {}
    for d in data_filtered:
        if d not in tiles:
            tiles[d] = 0
        tiles[d] += 1

    tiles_keys = list(tiles.keys())
    tiles_keys.sort()
    tiles = {i: tiles[i] for i in tiles_keys}

    plt.bar(list(tiles.keys()), tiles.values(), color="orange", width=10.0)
    plt.xlabel("max tile")
    plt.ylabel("# games")
    plt.savefig(os.path.join(ckpt_dir, f'{filename.split(".")[0]}.png'))
    plt.show()


def append_data(ckpt_dir="ckpt-#0-1_big-net", file_basename="d_reward.json"):
    data_0 = utils.load_data_from_file(os.path.join(ckpt_dir, f"0_{file_basename}"))
    data = utils.load_data_from_file(os.path.join(ckpt_dir, file_basename))

    for d in data:
        data_0.append(d)

    utils.save_data_in_file(data_0, os.path.join(ckpt_dir, file_basename))


def save(ckpt_dir):
    plot_and_save(ckpt_dir=ckpt_dir, filename="d_reward.json")
    plot_and_save(ckpt_dir=ckpt_dir, filename="deltas_policy.json")
    plot_and_save(ckpt_dir=ckpt_dir, filename="log_prob.json")
    plot_and_save(ckpt_dir=ckpt_dir, filename="losses_policy.json")
    plot_and_save(ckpt_dir=ckpt_dir, filename="losses_value.json")
    plot_and_save(ckpt_dir=ckpt_dir, filename="rewards.json")
    plot_and_save(ckpt_dir=ckpt_dir, filename="state_values.json")
    plot_and_save(ckpt_dir=ckpt_dir, filename="compute_percentiles_1000_rewards.json")
    plot_bar_and_save(ckpt_dir=ckpt_dir, filename="play_games_1000_ending_states.json")


def temp_percentile():
    dirs = ["ckpt-#0-2_big-net_replay_games", "ckpt-#0-3_big-net_replay_games", "ckpt-#0-4_big-net_replay_games"]
    data = []

    for mdir in dirs:
        for d in utils.load_data_from_file(os.path.join(mdir, "compute_percentiles_1000_rewards.json"))[1:]:
            data.append(d)

    plt.plot(data, ".", color="orange")
    plt.xlabel("# episodes")
    plt.savefig(os.path.join("C:\\Users\\Daniele\\Desktop\\images_aas", "ckpt-#0_big-net_replay_games_compute_percentiles_1000_rewards.png"))
    plt.show()


def temp_tiles():
    dirs = ["ckpt-#0-2_big-net_replay_games", "ckpt-#0-3_big-net_replay_games", "ckpt-#0-4_big-net_replay_games"]
    data = []

    for mdir in dirs:
        for d in utils.load_data_from_file(os.path.join(mdir, "play_games_1000_ending_states.json")):
            data.append(np.array(d[0]).flatten().max())

    tiles = {}
    for d in data:
        if d not in tiles:
            tiles[d] = 0
        tiles[d] += 1

    tiles_keys = list(tiles.keys())
    tiles_keys.sort()
    tiles = {i: tiles[i] for i in tiles_keys}

    plt.bar(list(tiles.keys()), tiles.values(), color="orange", width=10.0)
    plt.xlabel("max tile")
    plt.ylabel("# games")
    plt.savefig(os.path.join("C:\\Users\\Daniele\\Desktop\\images_aas", "ckpt-#0_big-net_replay_games_play_games_1000_ending_states.png"))
    plt.show()


def temp_rewards():
    dirs = ["ckpt-#0-2_big-net_replay_games", "ckpt-#0-3_big-net_replay_games", "ckpt-#0-4_big-net_replay_games"]
    data = []

    for mdir in dirs:
        for d in utils.load_data_from_file(os.path.join(mdir, "rewards.json"))[1:]:
            data.append(d)

    plt.plot(data, ".", color="orange")
    plt.xlabel("# episodes")
    plt.savefig(os.path.join("C:\\Users\\Daniele\\Desktop\\images_aas", "ckpt-#0_big-net_replay_games_rewards.png"))
    plt.show()


def main():
    # filter_ending_states(os.path.join("ckpt-#1-2_norm_huge-net_replay_games", "play_games_10000_ending_states.json"), 128)
    # plot_max_tile(os.path.join("ckpt-#1-2_norm_huge-net_replay_games", "play_games_10000_ending_states.json"))
    # plot_and_save()
    # save(ckpt_dir="ckpt-#0-4_big-net_replay_games")
    temp_percentile()
    temp_tiles()
    temp_rewards()


if __name__ == "__main__":
    main()
