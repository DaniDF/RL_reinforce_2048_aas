from reinforce_base_agent import ReinforceBaseAgent
from dqn_agent import DQNAgent
from data_logger import DataLogger
from policy_thread import PolicyThread
import utils

import time
import json


def loop_train(ckpt_num=1, ckpt_dir=".", n_episodes=10000, num_workers=5, alpha_policy=0.001, alpha_value=0.01,
               flag_plot_results=True):
    agent = ReinforceBaseAgent(input_shape=(4 * 4,), output_space=range(4))
    # agent = DQNAgent(input_shape=(4*4,), output_space=range(4))

    model_weight_file = utils.get_last_ckpt_name(ckpt_dir)
    if model_weight_file is not None:
        print("Using ckpt file: {}".format(model_weight_file))
        agent.load_model_weights(model_weight_file)

    data_logger = DataLogger(total_episodes=int(n_episodes / num_workers) * num_workers)

    for _ in range(ckpt_num):
        train(agent, data_logger, n_episodes=n_episodes, num_workers=num_workers, alpha_policy=alpha_policy,
              alpha_value=alpha_value)

        new_file = utils.get_next_ckpt_name()
        print("\nnew_file: {}".format(new_file))
        agent.save_model_weights(new_file)

        with open("losses_policy.json", "w") as file_out:
            json.dump([i.numpy().item() for i in data_logger.losses_policy], file_out)

        with open("losses_value.json", "w") as file_out:
            json.dump([i.numpy().item() for i in data_logger.losses_value], file_out)

        with open("deltas_policy.json", "w") as file_out:
            json.dump([i.numpy().item() for i in data_logger.deltas_policy], file_out)

        with open("rewards.json", "w") as file_out:
            json.dump([i.item() for i in data_logger.rewards], file_out)

        with open("d_reward.json", "w") as file_out:
            json.dump([i.numpy().item() for i in data_logger.d_reward], file_out)

        with open("state_values.json", "w") as file_out:
            json.dump([i.numpy().item() for i in data_logger.state_values], file_out)

        with open("log_prob.json", "w") as file_out:
            json.dump([i.numpy().item() for i in data_logger.log_prob], file_out)

    agent.save_model("model_policy.h5")

    if flag_plot_results:
        utils.plot_results(data_logger.losses_policy, data_logger.losses_value, data_logger.deltas_policy,
                           data_logger.d_reward, data_logger.state_values, data_logger.log_prob, data_logger.rewards)


def train(agent, data_logger, n_episodes=10000, num_workers=1, alpha_policy=1.0, alpha_value=1.0):
    workers = [PolicyThread(agent=agent,
                            data_logger=data_logger,
                            num_episodes=int(n_episodes / num_workers))
               for _ in range(num_workers)]

    print(time.strftime("%H:%M:%S", time.localtime()))
    data_logger.reset_episode_counter()
    for worker in workers:
        worker.start_worker_thread(alpha_policy=alpha_policy, alpha_value=alpha_value)

    for worker in workers:
        worker.join_worker_thread()
    print(time.strftime("\n%H:%M:%S", time.localtime()))


if __name__ == "__main__":
    loop_train(ckpt_num=5, n_episodes=int(20e3), num_workers=5, flag_plot_results=True)
