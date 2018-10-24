import numpy as np

def get_map_size(n_players):
    if n_players <= 3:
        return 16
    if n_players <= 13:
        return 32
    if n_players <= 20:
        return 64
        
    raise ValueError('n_players > 20 is not implemented yet')
