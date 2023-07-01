import numpy as np


class Game:
    def __init__(self, board_dim=4, max_tile=131072):
        self.observation_space = np.array([])
        self.board_dim = board_dim
        self.reset()

        self.action_space = range(4)

        self.game_score = 0
        self.MAX_TILE = max_tile

    def reset(self):
        self.observation_space = np.array([[0 for _ in range(self.board_dim)] for _ in range(self.board_dim)])
        self.__rand_new_cell__(np.random.choice([2, 4]))
        self.__rand_new_cell__(np.random.choice([2, 4]))
        return self.observation_space, "info"

    def step(self, action):
        reward = self.__move__(action)
        self.game_score += reward

        terminated = self.__rand_new_cell__()
        # terminated = self.__rand_new_cell__(np.random.choice([2, 4], p=[0.9, 0.1]))

        truncated = terminated

        return self.observation_space, reward, terminated, truncated, "info"

    def get_random_action(self):
        return np.random.choice(self.action_space)

    def get_max_tile(self):
        return np.max(self.observation_space)

    def close(self):
        return

    def __str__(self):
        result = ""

        for x in range(self.board_dim):
            for y in range(self.board_dim):
                result += f" {self.observation_space[x][y]:4d}"

            result += "\n"

        return result

    def __move__(self, action):
        self.observation_space = np.rot90(self.observation_space, k=action)
        reward = self.__splash_state__()
        self.observation_space = np.rot90(self.observation_space, k=4 - action)

        return reward

    def __splash_state__(self):
        reward = 0
        old_pre_c = -1
        for r in range(self.board_dim):
            for c in range(1, self.board_dim):
                pre_c = self.__pre_non_null__(r, c, old_pre_c)
                if c != pre_c:
                    if self.observation_space[r][pre_c] != 0:
                        old_pre_c = pre_c
                        reward += self.observation_space[r][pre_c] + self.observation_space[r][c]

                    self.observation_space[r][pre_c] += self.observation_space[r][c]
                    self.observation_space[r][c] = 0
            old_pre_c = -1

        return reward

    def __pre_non_null__(self, row, col, limit=-1):
        flag_done = False
        result = col
        for c in range(col - 1, limit, -1):
            if not flag_done and (
                    self.observation_space[row][c] == 0 or
                    self.observation_space[row][col] == self.observation_space[row][c]):
                result = c
            else:
                flag_done = True

        return result

    def __reward_board__(self):
        return np.amax(self.observation_space)

    def __rand_new_cell__(self, new_cell_value=2):
        empty_index = []

        for x in range(self.board_dim):
            for y in range(self.board_dim):
                if self.observation_space[x][y] == 0:
                    empty_index.append((x, y))

        flag_stop = len(empty_index) == 0

        if not flag_stop:
            rand_index = np.random.choice(range(len(empty_index)))
            self.observation_space[empty_index[rand_index][0]][empty_index[rand_index][1]] = new_cell_value

        return flag_stop
