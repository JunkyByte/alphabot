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
 
    def search(self, s, mapp, game, pipe, ask_predict, process_id, allow_move=False, alphabot=None):
        # logging.debug('Starting search')
        s_k = to_hash(s)
        if game.game_ended():
            # logging.debug('Game Ended during search')
            reward = game.reward
            game.reward = 0
            return -reward
        
        if s_k not in self.tree:
            # logging.debug('New state encountered')
            self.tree.append(s_k)
            if alphabot is None:
                ask_predict(process_id, s)
                raw_prediction = pipe.recv()
                policy, value = raw_prediction['policy'], raw_prediction['value']
            else:
                policy, value = alphabot.predict(s.astype(np.float32)[np.newaxis])
                policy = policy[0]
                value = value[0]
            
            policy = softmax(policy)[0]
            self.P[s_k], v = policy, value
            self.Q[s_k] = np.zeros((4))
            self.N[s_k] = np.zeros((4))
            return -v
        
        turn = s[..., -1].all() == 1  # get_turn(mapp)
        max_u, best_a = -float('inf'), -1
        if not allow_move:
            valid_actions = game.valid_actions(mapp, s, turn)
        else:
            valid_actions = [0, 1, 2, 3]

        if valid_actions == []:
            best_a = np.random.randint(0, 4)
            # return 1

        # logging.debug('Evaluating UCB')
        for a in valid_actions:  # The actions
            u = self.Q[s_k][a] + self.alpha * self.P[s_k][a] * np.sqrt(np.sum(self.N[s_k])) / (1 + self.N[s_k][a])
            if u > max_u or (u == max_u and np.random.random() > 0.5):
                max_u = u
                best_a = a
            # logging.debug('Action %d has a value of %f' % (a, u))
        
        a = best_a
        # logging.debug('\n ' + str(mapp))
        new_map = copy.deepcopy(mapp)
        new_map = game.step(new_map, s, a, turn)
        turn = 1 - turn  # get_turn(new_map)
        
        if turn == 0:  # We update the state
            sp = map_to_state(new_map, mapp, s, 0)
        else:
            # But we have to change the point of view of it!
            sp = copy.copy(s)
            sp[..., -1] = 1
        
        v = self.search(sp, new_map, game, pipe, ask_predict, process_id, allow_move, alphabot)
        
        self.Q[s_k][a] = (self.N[s_k][a] * self.Q[s_k][a] + v) / (self.N[s_k][a] + 1)
        self.N[s_k][a] += 1
        
        return -v


def softmax(z):
    z = z[np.newaxis]
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


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


def do_search(n, s, mapp, game, tree, pipe=None, ask_predict=None,
              process_id=None, alphabot=None, allow_move=False):
    for i in range(n):
        tree.search(s, mapp, game, pipe, ask_predict, process_id, alphabot=alphabot, allow_move=allow_move)

    x = tree.N[to_hash(s)]
    x = x / sum(x)
    return x


def simulate_game(steps, alpha, pipe, ask_predict, process_id, alphabot=None, eval_g=False):
    game = emulator.Game(2)
    mapp = game.reset()

    # tree = MCTS()
    # tree.alpha = alpha

    old_mapp = None
    turn = 0
    s = map_to_state(mapp, old_mapp, None, 0)
    old_mapp = copy.deepcopy(mapp)

    states = []
    policies = []
    while not game.game_ended():
        states.append(np.array(s))

        tree = MCTS()
        tree.alpha = alpha
        policy = do_search(steps, s, mapp, game, tree, pipe, ask_predict, process_id, alphabot=alphabot)
        if eval_g:
            choosen = np.argmax(policy)
        else:
            choosen = np.random.choice(4, p=policy)
        policies.append(np.array(policy))
        mapp = game.step(mapp, s, choosen, turn)

        # turn = get_turn(mapp)
        turn = 1 - turn
        if turn == 0:  # We update the state
            # logging.debug('Player 0 turn, updating the STATE')
            s = map_to_state(mapp, old_mapp, s, 0)  # TODO: Map to state
        else:
            # logging.debug('Player 1 turn, not updating the STATE')
            s[..., -1] = 1

        if turn == 0:
            old_mapp = np.array(mapp)

        logging.debug('Turn of %d Policy was %s Took action %s' % (1 - turn, np.round(policy, 2), game.dir_name[choosen]))
        printable_state = map_to_state(mapp, old_mapp, s, 0)
        printable_mapp = copy.copy(mapp)
        printable_mapp[np.where(printable_state[..., 1] == 1)] = 2
        printable_mapp[np.where(printable_state[..., 3] == 1)] = 3
        logging.debug('\n' + str(printable_mapp).replace('-1', '--'))
        
        if game.game_ended():
            winner = turn
            logging.debug('Game ended %d won' % (winner))
            train_steps = divide_states(winner, states, policies)
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
            value = 1 # As said below this is the last thing to understand (hopefully)
        else:  # These have been reverted TODO
            value = -1
        train_steps.append(TrainStep(state, value, policy))
        
        new_state = state[:, :, [2, 3, 0, 1, 4]]
        new_state[..., -1] = np.ones_like(new_state[..., -1]) - new_state[..., -1]
        train_steps.append(TrainStep(new_state, value, policy))
    
    return train_steps