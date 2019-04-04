import emulator
from mcts import MCTS, map_to_state, time_search
from keras.models import load_model
from keras.losses import mse
import tensorflow as tf
import random
import requests
import json
import time
import logging
import numpy as np
import copy
import os
from custom_layers import ZeroConv
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
alphabot = load_model('../alphabot_best.pickle', custom_objects={'ZeroConv' : ZeroConv, 'tf': tf})
url = 'https://lightningbot.tk/api/test'
bot_name = 'alphabot'  + str(random.randint(0, 99))
response = requests.get('/'.join([url, 'connect', bot_name]))  # Ask the server a token to play the game
connect = json.loads(response.text)  # Get the answer as a json to read it easily
assert connect['success'], connect


def move_bot(direction, current_turn):
    res = requests.get('/'.join([url, 'move', connect['token'], str(direction), str(current_turn)]))
    return json.loads(res.text)


def info_phase():
    res = requests.get('/'.join([url, 'info', connect['token']]))
    return json.loads(res.text)


def get_index(positions):
    for i, position in enumerate(positions):
        if position['pseudo'] == bot_name:
            return i
    assert False

def process_positions(positions, player_index, map_size):
    for i, position in enumerate(positions):
        if i == player_index:
            player_pos = (map_size - position['y'] - 1, position['x'])
        else:
            enemy_pos = (map_size - position['y'] - 1, position['x'])
    return player_pos, enemy_pos



def play_turn_enemy():
    return


def get_direction(turn, player_index):
    res = requests.get('/'.join([url, 'directions', connect['token'], str(turn)]))
    res = json.loads(res.text)
    if not 'directions' in res.keys():
        return -1, -1, res
    
    for i, action in enumerate(res['directions']):
        if i == player_index:
            player_dir = action['direction']
        else:
            enemy_dir = action['direction']

    return player_dir, enemy_dir, res


def play_game():
    logging.info('Connect Phase')
    time.sleep(connect['wait'] / 1000)  # Wait the right time to ask for information

    info = info_phase()
    logging.info('Info Phase')
    logging.info('Game Name: %s' % info['name'])
    logging.info('Bot Name: %s' % bot_name)
    logging.info('Map Size: %s' % info['dimensions'])
    map_size = info['dimensions']
    positions = info['positions']
    player_index = get_index(positions)
    starting_positions = list(process_positions(positions, player_index, map_size))
    logging.info('Starting Positions: %s' % starting_positions)

    # Setup Monte Carlo tree / Map
    game = emulator.Game(2, MAP_SIZE=16)
    old_mapp = None
    mapp = game.reset(starting_positions)
    mcts_tree = MCTS()
    mcts_tree.alpha = 1.1
    s = map_to_state(mapp, old_mapp, None, 0, INPUT_SIZE=16)
    old_mapp = copy.copy(mapp)

    running = True
    count_turn = 1
    time_to_move = info['wait'] / 1000  # Time in seconds to pick a move
    while running:
        policy, steps_done, v = time_search(time_to_move, s, mapp, game, mcts_tree, alphabot, INPUT_SIZE=16)
        action = np.argmax(policy)
        result = move_bot(action, count_turn)  # Move the bot and store the result of the request in a variable
        running = result['success']
        count_turn += 1

        logging.info('turn: %s' % count_turn)
        logging.info('steps_done: %s' % steps_done)
        logging.info('Value: %s' % v)
        logging.info('Policy: %s' % policy)

        time.sleep(max(0, result['wait'] / 1000))  # Wait until the next turn
        t = time.time()

        verify, action_enemy, response = get_direction(count_turn, player_index)
        if verify == -1:
            logging.info(response['description'])
            return 42

        mapp, tmp_head = game.step(mapp, s, action, 0, mcts=True)  # Process player turn
        mapp, _ = game.step(mapp, s, action_enemy, 1, mcts=True)  # Process enemy turn
        s = map_to_state(mapp, old_mapp, s, 0, 16, tmp_head)
        old_mapp = np.array(mapp)

        printable_map = s[..., 0] + s[..., 1] * 2 + s[..., 2]
        logging.info('\n' + str(printable_map))
        time_to_move = 1.1 - (time.time() - t)


if __name__ == '__main__':
    play_game()
