import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt


def print_perc(perc, max_char=20):
    sys.stdout.write("\r[")
    for count in range(0, int(perc * max_char)):
        sys.stdout.write("=")
    for count in range(0, max_char - int(perc * max_char)):
        sys.stdout.write(" ")
    sys.stdout.write("] {}%".format(perc * 100))


def update_perc(count_episode, n_episodes):
    perc = (count_episode+1) / n_episodes
    if perc * 1e4 % 10 == 0:
        print_perc(perc)


def get_last_ckpt_name(ckpt_path="."):
    try:
        path = os.path.join(ckpt_path, "checkpoint")
        with open(path, "r") as file_in:
            ckpt_last_file = os.path.join(ckpt_path, file_in.readline().split("\"")[1])
    except FileNotFoundError:
        ckpt_last_file = None

    return ckpt_last_file


def get_next_ckpt_name(ckpt_path="."):
    try:
        ckpt_last_file = get_last_ckpt_name(ckpt_path)
        if ckpt_last_file is not None:
            ckpt_last_num = int(ckpt_last_file[-8:-8 + 3])
            new_filename = f"cp-{ckpt_last_num + 1:03d}.ckpt"
        else:
            new_filename = "cp-001.ckpt"

    except FileExistsError:
        new_filename = "cp-001.ckpt"

    return new_filename


def plot_data(data, title, color):
    plt.plot(data, ".", color=color)
    plt.title(title)
    plt.show()


def plot_results(losses_policy, losses_value, deltas_policy, d_reward, state_values, log_prob, rewards):
    plot_data(losses_policy, "losses_policy", "blue")
    plot_data(losses_value, "losses_value", "brown")
    plot_data(deltas_policy, "deltas_policy", "orange")
    plot_data(d_reward, "d_reward", "black")
    plot_data(state_values, "state_values", "purple")
    plot_data(log_prob, "log_prob", "pink")
    plot_data(rewards, "rewards", "green")

    trend_reward = [np.average(rewards[index:index+10]) for index in range(0, len(rewards), 10)]

    plot_data(trend_reward, "trend_reward", "red")


def save_data_in_file(data, filename):
    with open(filename, "w") as file_out:
        json.dump(data, file_out)


def load_data_from_file(filename):
    with open(filename, "r") as file_in:
        result = json.load(file_in)

    return result
