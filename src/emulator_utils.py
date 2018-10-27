import numpy as np


def get_map_size(n_players):
    if n_players <= 3:
        return 16
    if n_players <= 13:
        return 32
    if n_players <= 20:
        return 64
        
    raise ValueError('n_players > 20 is not implemented yet')
    

def sum_tuple(t1, t2):
    return tuple(map(sum, zip(t1, t2)))
    

def mul_tuple(t1, t2):
    return tuple(map(lambda a, b : a * b, t1, t2))


def flip_game(history, v):
    assert type(v) == type(tuple())
    assert all(x in [-1, 1] for x in v) 
    # Flip history, each vector is a position and can be flipped based on a ([-1, 1], [-1, 1]) tuple
    for player, positions in history.items(): # Iterate through each player history
        for idx in range(len(positions)): # Iterate through each movement the player has taken
            # Flip the position based on v (our flip vectors)
            history[player][idx] = mul_tuple(history[player][idx], v)
    return history
    
def h_to_map(history, step):
    assert type(step) == type(int())
    # Now let's compute the map at instant step
    n_players = len(history.keys()) # Infer number of players
    MAP_SIZE = get_map_size(n_players)
    map = np.full((MAP_SIZE, MAP_SIZE), fill_value=-1, dtype=np.int) # -1 is null map and should be globally def.
    
    for player, positions in history.items():
        for i in range(step + 1): # For each step up to step
            try:
                map[history[player][i]] = player
            except:
                break
    return map


def delta_move(old_pos, new_pos, dir_vector, dir_names): # Two consecutive positions and the dir_vector used
    # Computes what action a player has taken based on delta position between two steps
    delta = sum_tuple(new_pos, -old_pos)
    inv_dir_vector = {v : k for k, v in dir_vector.items()}
    inv_dir_names = {v : k for k, v in dir_names.items()}
    
    # Be sure the delta is valid inside the movement space
    assert delta in inv_dir_vector.keys()
    assert inv_dir_vector[delta] in inv_dir_names.keys()
    
    action = inv_dir_names[inv_dir_vector[delta]]
    return action
