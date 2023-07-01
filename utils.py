import sys
import os

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
            # new_filename = "cp-00{}.ckpt".format(ckpt_last_num + 1)
            new_filename = f"cp-{ckpt_last_num + 1:03d}.ckpt"
        else:
            new_filename = "cp-001.ckpt"

    except FileExistsError:
        new_filename = "cp-001.ckpt"

    return new_filename


def plot_results(losses_policy, losses_value, deltas_policy, d_reward, state_values, log_prob, rewards):
    plt.plot(losses_policy, ".", color="blue")
    plt.title("losses_policy")
    plt.show()

    plt.plot(losses_value, ".", color="brown")
    plt.title("losses_policy")
    plt.show()

    plt.plot(deltas_policy, ".", color="orange")
    plt.title("losses_value")
    plt.show()

    plt.plot(d_reward, ".", color="black")
    plt.title("d_reward")
    plt.show()

    plt.plot(state_values, ".", color="purple")
    plt.title("state_values")
    plt.show()

    plt.plot(log_prob, ".", color="pink")
    plt.title("log_prob")
    plt.show()

    plt.plot(rewards, ".", color="green")
    plt.title("rewards")
    plt.show()

    trend_reward = [np.average(rewards[index:index+10]) for index in range(0, len(rewards), 10)]

    plt.plot(trend_reward, ".", color="red")
    plt.title("trend_reward")
    plt.show()
