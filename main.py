import os.path

from reinforce_base_agent import ReinforceBaseAgent
from dqn_agent import DQNAgent
from data_logger import DataLogger
from policy_thread import PolicyThread
from replay_games import ReplayGames

import utils
import time


def loop_train(ckpt_num=1, ckpt_dir=".", n_episodes=10000, num_workers=5, alpha_policy=0.001, alpha_value=0.01,
               flag_plot_results=True, save_stats=True, replay_games_file=None, replay_games_proba=0.0):
    agent = ReinforceBaseAgent(input_shape=(4 * 4,), output_space=range(4))
    # agent = DQNAgent(input_shape=(4*4,), output_space=range(4))

    model_weight_file = utils.get_last_ckpt_name(ckpt_dir)
    if model_weight_file is not None:
        print("Using ckpt file: {}".format(model_weight_file))
        agent.load_model_weights(model_weight_file)

    data_logger = DataLogger(total_episodes=int(n_episodes / num_workers) * num_workers)
    replay_games = None
    if replay_games_file is not None:
        replay_games = ReplayGames(os.path.join(ckpt_dir, replay_games_file))

    for _ in range(ckpt_num):
        train(agent, data_logger, n_episodes=n_episodes, num_workers=num_workers, alpha_policy=alpha_policy,
              alpha_value=alpha_value, replay_games=replay_games, replay_games_proba=replay_games_proba)

        new_file = utils.get_next_ckpt_name(ckpt_dir)
        print("\nnew_file: {}".format(new_file))
        agent.save_model_weights(new_file)

        if save_stats:
            utils.save_data_in_file([i.numpy().item() for i in data_logger.losses_policy], "losses_policy.json")
            utils.save_data_in_file([i.numpy().item() for i in data_logger.losses_value], "losses_value.json")
            utils.save_data_in_file([i.numpy().item() for i in data_logger.deltas_policy], "deltas_policy.json")
            utils.save_data_in_file([i.item() for i in data_logger.rewards], "rewards.json")
            utils.save_data_in_file([i.numpy().item() for i in data_logger.d_reward], "d_reward.json")
            utils.save_data_in_file([i.numpy().item() for i in data_logger.state_values], "state_values.json")
            utils.save_data_in_file([i.numpy().item() for i in data_logger.log_prob], "log_prob.json")

    if flag_plot_results:
        utils.plot_results(data_logger.losses_policy, data_logger.losses_value, data_logger.deltas_policy,
                           data_logger.d_reward, data_logger.state_values, data_logger.log_prob, data_logger.rewards)


def train(agent, data_logger, n_episodes=10000, num_workers=1, alpha_policy=1.0, alpha_value=1.0, replay_games=None,
          replay_games_proba=0.0):
    workers = [PolicyThread(agent=agent,
                            data_logger=data_logger,
                            num_episodes=int(n_episodes / num_workers),
                            replay_games=replay_games,
                            replay_games_proba=replay_games_proba)
               for _ in range(num_workers)]

    print(time.strftime("%H:%M:%S", time.localtime()))
    data_logger.reset_episode_counter()
    for worker in workers:
        worker.start_worker_thread(alpha_policy=alpha_policy, alpha_value=alpha_value)

    for worker in workers:
        worker.join_worker_thread()
    print(time.strftime("\n%H:%M:%S", time.localtime()))


if __name__ == "__main__":
    # loop_train(ckpt_num=3, n_episodes=int(20e3), num_workers=5, flag_plot_results=False, save_states=True, replay_games_file=None)
    loop_train(ckpt_num=1, n_episodes=int(1), num_workers=1, flag_plot_results=False, save_stats=False,
               ckpt_dir="ckpt-#0-4_norm_huge-net_replay_games", replay_games_file="play_games_1000_ending_states.json",
               replay_games_proba=0.33)
