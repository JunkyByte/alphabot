import numpy as np
import emulator_utils
import emulator_vis
from copy import copy
null_map = -1

# Define Map Params


class Game():
    def __init__(self, n_players):
        np.set_printoptions(threshold=np.nan)
        self.n_players = n_players
        self.MAP_SIZE = emulator_utils.get_map_size(self.n_players)
        self.dir_name = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
        self.dir_vect = {'right': (0, 1), 'down': (1, 0), 'left': (0, -1), 'up': (-1, 0)}
        self.reward = 0

    def nobody_around(self, position, mapp):
        x, y = position
        return mapp[x, y] == mapp[x, y + 1] == mapp[x, y - 1] == mapp[x + 1, y] == mapp[x + 1, y + 1] == \
            mapp[x + 1, y - 1] == mapp[x - 1, y] == mapp[x - 1, y + 1] == mapp[x - 1, y - 1]

    def reset(self):
        mapp = np.full((self.MAP_SIZE, self.MAP_SIZE), fill_value=null_map, dtype=np.int)

        def random_position(mapp):
            while True:
                position = tuple(np.random.randint(0, self.MAP_SIZE - 2, size=2))
                if mapp[position] == null_map and self.nobody_around(position, mapp):
                    return position

        for p in range(self.n_players):
            position = random_position(mapp)
            mapp[position] = p

        return mapp

    def sanitize_positions(self, pos):
        def r_div(n):
            return n % self.MAP_SIZE
        return (r_div(pos[0]), r_div(pos[1]))

    def game_ended(self):
        return self.reward != 0

    def get_head(self, state, turn):
        channel_index = 2 * turn + 1  # This is the channel where the head is located
        head_channel = state[..., channel_index]
        return np.where(head_channel == 1)  # Where's the head?

    def step(self, p, state, action):
        mapp = copy(p)

        # We are getting the state here. last channel of it is the turn
        turn = int(state[..., -1].all() == 1)
        head_position = self.get_head(state, turn)

        def compute_direction(direction):
            return self.dir_vect[self.dir_name[direction]]

        direction = compute_direction(action)
        new_position = self.sanitize_positions(emulator_utils.sum_tuple(head_position, direction))

        point = mapp[new_position]
        if point != null_map:
            self.reward = -1  # Game ended
        else:  # We have to update the mapp
            mapp[new_position] = turn

        return mapp
