import utils

import numpy as np


class ReplayGames:
    def __init__(self, replay_filename, clean_cells=3):
        self.replay_buffer = np.array(utils.load_data_from_file(replay_filename))

        for index, game in enumerate(self.replay_buffer):
            shape = game.shape

            count_row = 0
            count_col = 0

            flat_game = game.flatten()
            arg_flat_game = []

            for cell in flat_game:
                arg_flat_game.append((count_row, count_col, cell))

                count_col += 1
                if count_col >= shape[1]:
                    count_col = 0
                    count_row += 1

            arg_flat_game.sort(key=lambda x: x[2])
            for cell_index in range(clean_cells):
                pos = arg_flat_game[cell_index]
                self.replay_buffer[index][pos[0]][pos[1]] = 0

    def sample_game(self):
        result_index = np.random.randint(len(self.replay_buffer))
        return self.replay_buffer[result_index]
