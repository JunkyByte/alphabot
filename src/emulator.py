import numpy as np
import emulator_utils

null_map = -1

# Define Map Params
class Game():
    def __init__(self, n_players):
        np.set_printoptions(threshold=np.nan)
        self.n_players = n_players
        self.players_alive = np.ones((self.n_players))
        self.MAP_SIZE = emulator_utils.get_map_size(self.n_players)
        self.map = np.full((self.MAP_SIZE, self.MAP_SIZE), fill_value=null_map, dtype=np.int)
        self.history = self.init_players()
        self.dir_name = {0 : 'right', 1 : 'down', 2 : 'left', 3 : 'up'}
        self.dir_vect = {'right' : (0, 1), 'down' : (1, 0), 'left' : (0, -1), 'up' : (-1, 0)}

    def init_players(self):
        def random_position():
            while True:
                position = tuple(np.random.randint(0, self.MAP_SIZE - 2, size=2))
                if self.map[position] == null_map:
                    return position

        history = {}
        for p in range(self.n_players):
            position = random_position()
            history[p] = []
            history[p].append(position)
            self.map[history[p][-1]] = p
        return history

    def reset(self):
        self.__init__(self.n_players)

    def count_alive(self):
        return sum(self.players_alive == 1)
        
    def check_game_end(self):
        if self.count_alive() == 1:
            return True
        return False
        
    def assign_win(self, rewards):
        assert self.count_alive() == 1
        rewards[np.where(rewards >= 0)] += 1 # Assign reward to the alive player
        return rewards

    def sanitize_positions(self, idx):
        def r_div(n):
            return n % self.MAP_SIZE

        x, y = self.history[idx][-1]
        self.history[idx][-1] = (r_div(x), r_div(y))

    def step(self, alive_actions):
        assert len(alive_actions) == self.count_alive() # Verify that there's one action per player alive
        assert min(alive_actions) >= -1 and max(alive_actions) <= 3 # dead player action is -1, [0;3] allowed movements
        def compute_direction(direction):
            return self.dir_vect[self.dir_name[direction]]

        # Create reward vector
        reward = np.zeros((self.count_alive()))

        # Convert action to same length as total players
        actions = np.full((self.n_players), fill_value=-1)
        for idx, point in enumerate(np.where(self.players_alive == 1)):
            actions[point] = alive_actions[idx]

        for idx, action in enumerate(actions): # Iterate through each player action
            if action == -1: # Dead players
                continue

            # Compute new position for each player alive
            direction = compute_direction(action)
            self.history[idx].append(emulator_utils.sum_tuple(self.history[idx][-1], direction))
            self.sanitize_positions(idx)

            # verify if any player died
            if self.map[self.history[idx][-1]] != null_map: # Default map value is empty
                self.players_alive[idx] = 0 # Ouch! idx died!
                reward[idx] += -1 # Negative reward to idx for dying
                if self.map[self.history[idx][-1]] != idx: # Check if is a suicide
                    reward[self.map[self.history[idx][-1]]] += 0.5 # Positive reward for the killer
                if self.check_game_end(): # If game ended just break the updating
                    break

                continue # Go to next player as this one doesn't need to be rendered
                
            # Our true warriors get updated on the map!
            self.map[self.history[idx][-1]] = idx
            
        game_ended = self.check_game_end()
        if game_ended:
            reward = self.assign_win(reward) # Will return the reward vector with added win
        
        # Returns State, list of alive, number of alive, reward for each player (that was alive), game end
        return self.map, self.players_alive, self.count_alive(), reward, game_ended


    def show(self):
        print(self.map)
        print(''.join('-' for _ in range(self.MAP_SIZE * 3)))


    def replay_game(self):
        # TODO: Must return array of 'frames' which are the map step by step (Use self.history)
        return
    
    def flip_game(self, v):
        # Flips the game based on v and returns the flipped history
        return emulator_utils.flip_game(self.history, v) 


if __name__ == '__main__':
    p = 3
    g = Game(p)
    g.show()
    n_alive = p

    for i in range(25):
        print('Step', i)
        actions = np.random.randint(0, 3, size=n_alive) # Actions of alive bots!
        map, p_alive, n_alive, rewards, game_end = g.step(actions)
        print(p_alive)
        print(rewards)
        if game_end:
            break
            
    g.show()
    g.flip_game((-1, 1))
