import copy
import numpy as np
import emulator
import time
FULL_SIZE = (16, 16, 5)


class MCTS():
    def __init__(self):
        self.tree = []
        self.P = {}
        self.Q = {}
        self.N = {}
        self.alpha = 0.8

    def search(self, s, mapp, game, pipe, ask_predict, process_id, INPUT_SIZE,
               allow_move=False, alphabot=None, head_pos=None):
        # logging.debug('Starting search')
        s_k = to_hash(s)
        if game.game_ended():
            # logging.debug('Game Ended during search')
            reward = game.reward
            game.reward = 0
            return -reward

        turn = s[..., -1].all() == 1
        if s_k not in self.tree:
            # logging.debug('New state encountered')
            self.tree.append(s_k)
            if isinstance(alphabot, (str, bool)) or alphabot is None:  # Revert
                ask_predict(process_id, s, alphabot)
                raw_prediction = pipe.recv()
                policy, value = raw_prediction['policy'], raw_prediction['value']
            else:
                policy, value = alphabot.predict(s[np.newaxis].astype(np.float32))
                policy = policy[0]
                value = value[0]

            valid_actions = game.valid_actions(mapp, s, turn)
            if len(valid_actions) < 4:
                missing_idx = [v for v in [0, 1, 2, 3] if v not in valid_actions]
                policy[missing_idx] = 0

            if sum(policy) > 0:
                policy = policy / sum(policy)
            else:
                policy = np.ones((4)) / 4

            self.P[s_k], v = policy, value
            self.Q[s_k] = np.ones((4)) * -100
            self.N[s_k] = np.zeros((4))
            return -v
        
        max_u, best_a = -float('inf'), -1
        if not allow_move:
            valid_actions = game.valid_actions(mapp, s, turn)
        else:
            valid_actions = [0, 1, 2, 3]

        for a in valid_actions:  # The actions
            if self.Q[s_k][a] != -100:
                u = self.Q[s_k][a] + self.alpha * self.P[s_k][a] * \
                    np.sqrt(np.sum(self.N[s_k])) / (1 + self.N[s_k][a])
            else:
                u = self.alpha * self.P[s_k][a] * np.sqrt(np.sum(self.N[s_k]) + 1e-7)

            if u > max_u or (u == max_u and np.random.random() > 0.5):
                max_u = u
                best_a = a
        
        if valid_actions == []:
            a = np.random.randint(0, 4)
        else:
            a = best_a

        new_map = copy.deepcopy(mapp)
        new_map, tmp_head = game.step(new_map, s, a, turn, mcts=True)
        turn = 1 - turn
        
        if turn == 0:  # We update the state
            sp = map_to_state(new_map, mapp, s, 0, INPUT_SIZE, head_pos)
        else:
            # But we have to change the point of view of it!
            sp = copy.copy(s)
            head_pos = tmp_head
            sp[..., -1] = 1
        
        v = self.search(sp, new_map, game, pipe, ask_predict, process_id, INPUT_SIZE,
                        allow_move, alphabot, head_pos)
        
        if self.Q[s_k][a] == -100:
            self.Q[s_k][a] = v
        else:
            self.Q[s_k][a] = (self.N[s_k][a] * self.Q[s_k][a] + v) / (self.N[s_k][a] + 1)
        self.N[s_k][a] += 1
        
        return -v


def to_hash(state):
    return hash(state.tostring())


def map_to_state(gmap, gmap_old, state, turn, INPUT_SIZE, head_pos=None):
    if type(gmap_old) != np.ndarray:
        gmap_old = np.full_like(gmap, -1)

    states = np.empty(FULL_SIZE, dtype=np.int)

    states = process_map(gmap, gmap_old, state, turn, INPUT_SIZE, head_pos)
    return states


def process_map(gmap, gmap_old, state, idx, INPUT_SIZE, head_pos=None):
    pov_0 = np.zeros((*FULL_SIZE[:2], 1), dtype=np.int)
    pov_0_head = np.zeros((*FULL_SIZE[:2], 1), dtype=np.int)
    pov_1 = np.zeros((*FULL_SIZE[:2], 1), dtype=np.int)
    pov_1_head = np.zeros((*FULL_SIZE[:2], 1), dtype=np.int)

    pov_0[np.where(gmap == 0)] = 1  # Set to 1 where player 0 is
    pov_0_head[np.where(gmap_old == 0)] = 1

    pov_1[np.where(gmap == 1)] = 1  # Set to  1 where player 1 is
    pov_1_head[np.where(gmap_old == 1)] = 1

    pov_0_head = pov_0 - pov_0_head
    pov_1_head = pov_1 - pov_1_head

    if sum(sum(pov_0_head)) == 0:
        if head_pos is not None:
            pov_0_head[head_pos] = 1
        else:
            pov_0_head = np.expand_dims(state[..., 0], axis=-1)

    if sum(sum(pov_1_head)) == 0:
        pov_1_head = np.expand_dims(state[..., 1], axis=-1)

    turn_m = np.full((*FULL_SIZE[:2], 1), dtype=np.int, fill_value=idx)
    occupied = pov_0 + pov_1

    mask = np.zeros((*FULL_SIZE[:2], 1), dtype=np.int)
    mask_ones = np.ones((INPUT_SIZE, INPUT_SIZE, 1), dtype=np.int)
    mask[:INPUT_SIZE, :INPUT_SIZE] = mask_ones

    return np.concatenate([pov_0_head, pov_1_head, occupied, mask, turn_m], axis=2)


def do_search(n, s, mapp, game, tree, INPUT_SIZE, pipe=None, ask_predict=None,
              process_id=None, alphabot=None, allow_move=False, tau=1):
    for i in range(n):
        tree.search(s, mapp, game, pipe, ask_predict, process_id, INPUT_SIZE,
                    alphabot=alphabot, allow_move=allow_move)

    x = tree.N[to_hash(s)]
    x = 0.75 * x + 0.25 * np.random.dirichlet(0.3 * np.ones(len(x)))
    x = np.power(x, 1 / tau) / sum(np.power(x, 1 / tau))
    return x


def time_search(move_time, s, mapp, game, tree, alphabot, INPUT_SIZE=16):
    t = time.time()
    counter = 0
    while time.time() - t < move_time:
        tree.search(s, mapp, game, None, None, None, INPUT_SIZE,
                    alphabot=alphabot, allow_move=False)
        counter += 1

    x = tree.N[to_hash(s)]
    x = x / sum(x)
    _, value = alphabot.predict(s[np.newaxis].astype(np.float32))
    value = value[0]

    return x, counter, value


def simulate_game(steps, alpha, INPUT_SIZE, pipe=None, ask_predict=None, process_id=None,
                  alphabot=None, return_state=False, count_turn=0):

    if isinstance(steps, list):
        steps_long = steps[1]
        steps = steps[0]
    else:
        steps_long = steps

    # Game Setup
    game = emulator.Game(2, INPUT_SIZE)
    mapp = game.reset()
    old_mapp = None
    turn = 0
    s = map_to_state(mapp, old_mapp, None, 0, INPUT_SIZE)
    old_mapp = copy.deepcopy(mapp)
    head = None

    states = []
    policies = []
    tau = 1
    while not game.game_ended():
        tree = MCTS()
        tree.alpha = alpha

        count_turn += 1 / 2
        states.append(np.array(s))
        
        # Callbacks
        if count_turn > INPUT_SIZE:
            tau = 1e-1

        if count_turn > INPUT_SIZE * 2:
            steps = steps_long

        policy = do_search(steps, s, mapp, game, tree, INPUT_SIZE, pipe, ask_predict, process_id,
                           alphabot=alphabot, tau=tau)

        choosen = np.argmax(policy)
        
        policies.append(np.array(policy))
        mapp, tmp_head = game.step(mapp, s, choosen, turn, mcts=True)

        turn = 1 - turn
        if turn == 0:  # We update the state
            s = map_to_state(mapp, old_mapp, s, 0, INPUT_SIZE, head)
        else:
            head = tmp_head
            s[..., -1] = 1

        if turn == 0:
            old_mapp = np.array(mapp)

        if game.game_ended():
            winner = turn
            # logging.debug('Game ended %d won' % (winner))
            if return_state:
                return states

            train_steps = divide_states(winner, states, policies, INPUT_SIZE)
            return train_steps


class TrainStep():
    def __init__(self, state, value, policy, map_size):
        self.state = state
        self.value = value
        self.policy = policy
        self.map_size = map_size


def divide_states(winner, states, policies, INPUT_SIZE):
    train_steps = []

    for state, policy in zip(states, policies):
        if state[..., -1].all() == winner:
            value = 1
        else:
            value = -1
        train_steps.append(TrainStep(state, value, policy, INPUT_SIZE))

        new_state = state[:, :, [1, 0, 2, 3, 4]]

        new_state[..., -1] = np.ones_like(new_state[..., -1]) - new_state[..., -1]
        train_steps.append(TrainStep(new_state, value, policy, INPUT_SIZE))

    return train_steps
