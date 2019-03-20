import numpy as np
from mcts import TrainStep


def policy_rot90(policy, k=1):
    k = k % 4
    policy = np.array(policy)
    for i in range(k):
        policy = policy[..., [1, 2, 3, 0]]

    return policy


def policy_flip(policy, vert=False):
    policy = np.array(policy)
    if vert:
        return policy[..., [0, 3, 2, 1]]

    return policy[..., [2, 1, 0, 3]]


def state_flip(state, MAP_SIZE, vert=False):
    state = np.array(state)
  
    if vert:
        p = state[:, :MAP_SIZE]
        state[:, :MAP_SIZE] = p[:, ::-1]
        return state
    p = state[:, :MAP_SIZE, :MAP_SIZE]
    state[:, :MAP_SIZE, :MAP_SIZE] = p[:, :, ::-1]
    return state


def apply_simmetries(train_steps, MAP_SIZE):
    # 90;180:270 degrees rotations
    # Flips

    t_s = []
    t_p = []
    value = []
    map_sizes = []
    for step in train_steps:
        t_s.append(step.state)
        t_p.append(step.policy)
        value.append(step.value)
        map_sizes.append(step.map_size)

    for i in range(1, 4):
        for j in range(0, 2):
            state = np.array(t_s)
            p = np.rot90(state[:, :MAP_SIZE, :MAP_SIZE, :], k=i, axes=(1, 2))
            state[:, :MAP_SIZE, :MAP_SIZE, :] = p
            policy = policy_rot90(t_p, k=i)

            if j == 0:  # Horizontal flip
                state = state_flip(state, MAP_SIZE, vert=False)
                policy = policy_flip(policy, vert=False)
            elif j == 1:  # Vertical flip
                state = state_flip(state, MAP_SIZE, vert=True)
                policy = policy_flip(policy, vert=True)

            train_steps.extend([TrainStep(s, v, p, m) for s, p, v, m in zip(state, policy, value, map_sizes)])

    return train_steps
