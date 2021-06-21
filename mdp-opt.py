import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

def step(action, state, world_size):
    action = np.array(action)
    state = np.array(state)
    new_state = action + state

    if (state[0] == 0 and state[1] == 0) or (state[0] == world_size-1 and state[1] == world_size-1):
        return state, 1
    elif (new_state[0] == 0 and new_state[1] == 0) or (new_state[0] == world_size-1 and new_state[1] == world_size-1):
        return new_state, 0

    if new_state[0] < 0 or new_state[0] >= world_size or new_state[1] < 0 or new_state[1] >= world_size:
        return state, -2
    else:
        return new_state, -1

def set_env(world_size, terms):
    env = np.zeros((world_size, world_size))
    for x, y in terms:
        env[x, y] = 1.

    return env

def main() :
    world_size = 5
    discount_factor = 0.5
    pi = 0.25
    terms = [[0, 0], [4, 4]]
    actions = [[0, 1], [0, -1], [-1, 0], [1, 0]]
    value = set_env(world_size, terms)

    print(value)

    fig = plt.figure()

    while True:
        new_value = np.zeros((world_size, world_size))

        for x in range(world_size):
            for y in range(world_size):
                nv = list()
                for action in actions:
                    new_state, reward = step(action, [y, x], world_size)
                    nv.append(pi * (reward + discount_factor * value[new_state[0], new_state[1]]))
                new_value[y, x] = np.max(nv)
        if np.sum(np.abs(value-new_value)) < 1e-4:
            break

        value = new_value
        print()
        print(value)
        plt.cla()
        plt.imshow(value, cmap='jet')
        plt.colorbar()
        plt.draw()
        plt.pause(1)
    print(value)
    plt.show()

if __name__ == '__main__':
    main()