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


def state_flip(state, vert=False):
    state = np.array(state)

    if vert:
        return state[:, ::-1]
    return state[:, :, ::-1]


def apply_simmetries(train_steps):
    # 90;180:270 degrees rotations
    # Flips

    t_s = []
    t_p = []
    value = []
    for step in train_steps:
        t_s.append(step.state)
        t_p.append(step.policy)
        value.append(step.value)

    for i in range(1, 4):
        for j in range(0, 2):
            state = np.rot90(t_s, k=i, axes=(1, 2))
            policy = policy_rot90(t_p, k=i)

            if j == 0:  # Horizontal flip
                state = state_flip(state, vert=False)
                policy = policy_flip(policy, vert=False)
            elif j == 1:  # Vertical flip
                state = state_flip(state, vert=True)
                policy = policy_flip(policy, vert=True)

            train_steps.extend([TrainStep(s, v, p) for s, p, v in zip(state, policy, value)])

    return train_steps
