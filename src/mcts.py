import logging
import copy
import numpy as np
import emulator

INPUT_SIZE = (16, 16, 5)  # Map size fixed to 16x16 (2 to 3 players)


class MCTS():
    def __init__(self):
        self.tree = []
        self.P = {}
        self.Q = {}
        self.N = {}
        self.alpha = 0.8
 
    def search(self, s, mapp, game, pipe, ask_predict, process_id):
        logging.debug('Starting search')
        s_k = to_hash(s)
        if game.game_ended():
            logging.debug('Game Ended during search')
            game.reward = 0
            return -1
        
        if s_k not in self.tree:
            logging.debug('New state encountered')
            self.tree.append(s_k)
            ask_predict(process_id, s)
            raw_prediction = pipe.recv()
            policy, value = raw_prediction['policy'], raw_prediction['value']
            self.P[s_k], v = policy[0], value[0]
            self.Q[s_k] = np.zeros((4))
            self.N[s_k] = np.zeros((4))
            return -v
        
        max_u, best_a = -float('inf'), -1
        logging.debug('Evaluating UCB')
        for a in range(4):  # The actions
            u = self.Q[s_k][a] + self.alpha * self.P[s_k][a] * np.sqrt(np.sum(self.N[s_k]) / (1 + self.N[s_k][a]))
            if u > max_u or (u == max_u and np.random.random() > 0.5):
                max_u = u
                best_a = a
            logging.debug('Action %d has a value of %f' % (a, u))
        a = best_a
        
        turn = get_turn(mapp)
        logging.debug('\n ' + str(mapp))
        new_map = copy.deepcopy(mapp)
        new_map = game.step(new_map, s, a, turn)
        turn = get_turn(new_map)
        
        logging.debug(game.reward)
        logging.debug('New map after move, now is turn of %d' % turn)
        logging.debug('\n' + str(new_map))
        
        if turn == 0:  # We update the state
            logging.debug('Player 0 turn, updating the map')
            sp = map_to_state(new_map, mapp, s, 0)  # TODO: Map to state
        else:
            logging.debug('Player 1 turn, not updating the map')
            # But we have to change the point of view of it!
            sp = copy.copy(s)
            sp[..., -1] = 1
        
        v = self.search(sp, new_map, game, pipe, ask_predict, process_id)
        
        self.Q[s_k][a] = (self.N[s_k][a] * self.Q[s_k][a] + v) / (self.N[s_k][a] + 1)
        self.N[s_k][a] += 1
        
        return -v


def to_hash(state):
    return hash(state.tostring())


def get_turn(x):
    idx, count = np.unique(x, return_counts=True)
    idx, count = idx[1:], count[1:]
    if count[0] == count[1]:
        return 0
    
    return idx[np.argmin(count)]


def map_to_state(gmap, gmap_old, state, turn):
    if type(gmap_old) != np.ndarray:
        gmap_old = np.full_like(gmap, -1)

    states = np.empty(INPUT_SIZE, dtype=np.int)

    states = process_map(gmap, gmap_old, state, turn)
    return states


def process_map(gmap, gmap_old, state, idx):
    pov_0 = np.zeros((*INPUT_SIZE[:2], 1), dtype=np.int)
    pov_0_last = np.zeros((*INPUT_SIZE[:2], 1), dtype=np.int)
    pov_1 = np.zeros((*INPUT_SIZE[:2], 1), dtype=np.int)
    pov_1_last = np.zeros((*INPUT_SIZE[:2], 1), dtype=np.int)

    pov_0[np.where(gmap == 0)] = 1  # Set to 1 where player 0 is
    pov_0_last[np.where(gmap_old == 0)] = 1

    pov_1[np.where(gmap == 1)] = 1  # Set to  1 where player 1 is
    pov_1_last[np.where(gmap_old == 1)] = 1

    pov_0_last = pov_0 - pov_0_last
    pov_1_last = pov_1 - pov_1_last

    if sum(sum(pov_0_last)) == 0:
        pov_0_last = np.expand_dims(state[..., 1], axis=-1)

    if sum(sum(pov_1_last)) == 0:
        pov_1_last = np.expand_dims(state[..., 3], axis=-1)

    turn_m = np.full((*INPUT_SIZE[:2], 1), dtype=np.int, fill_value=idx)

    return np.concatenate([pov_0, pov_0_last, pov_1, pov_1_last, turn_m], axis=2)


def do_search(n, s, mapp, game, tree, pipe, ask_predict, process_id):
    for i in range(n):
        tree.search(s, mapp, game, pipe, ask_predict, process_id)

    x = tree.N[to_hash(s)]
    x = x / max(sum(x), 1e-7)
    return x


def simulate_game(steps, alpha, pipe, ask_predict, process_id):
    game = emulator.Game(2)
    mapp = game.reset()

    tree = MCTS()
    tree.alpha = alpha

    old_mapp = None
    turn = 0
    s = map_to_state(mapp, old_mapp, None, 0)
    old_mapp = copy.deepcopy(mapp)

    states = []
    policies = []
    while not game.game_ended():
        states.append(np.array(s))
        policy = do_search(steps, s, mapp, game, tree, pipe, ask_predict, process_id)
        choosen = np.argmax(policy)
        policies.append(np.array(policy))
        mapp = game.step(mapp, s, choosen, turn)

        turn = get_turn(mapp)
        if turn == 0:  # We update the state
            logging.debug('Player 0 turn, updating the STATE')
            s = map_to_state(mapp, old_mapp, s, 0)  # TODO: Map to state
        else:
            logging.debug('Player 1 turn, not updating the STATE')
            s[..., -1] = 1

        if turn == 0:
            old_mapp = np.array(mapp)

        if game.game_ended():
            train_steps = divide_states(int(not turn), states, policies)
            return train_steps


class TrainStep():
    def __init__(self, state, value, policy):
        self.state = state
        self.value = value
        self.policy = policy


def divide_states(winner, states, policies):
    train_steps = []

    for state, policy in zip(states, policies):
        if state[..., -1].all() == winner:
            value = 1
        else:
            value = -1
        train_steps.append(TrainStep(state, value, policy))

    return train_steps
