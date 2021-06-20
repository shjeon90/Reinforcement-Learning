import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from collections import defaultdict

# agent가 어떤 state에 도달했을 때, reward를 반환하는 역할
class Environment:
    def __init__(self, world_size):
        self.world_size = world_size
        self.terms = [(0, 0), (world_size-1, world_size-1)]

    def _is_oob(self, state):
        x, y = state
        if x < 0 or x >= self.world_size: return True
        elif y < 0 or y >= self.world_size: return True
        else: return False

    def step(self, state, action):
        x, y = state

        if action == 0:  # up
            y -= 1
        elif action == 1:  # down
            y += 1
        elif action == 2:  # left
            x -= 1
        else:  # right
            x += 1

        if self._is_oob((x, y)):
            next_state = state
            return -2., next_state, False
        else:
            next_state = (x, y)
            if next_state in self.terms:
                return 0., next_state, True
            else:
                return -1., next_state, False

    def reset(self):
        while True:
            x, y = np.random.randint(self.world_size), np.random.randint(self.world_size)
            if (x, y) not in self.terms:
                break
        return x, y

# action-value를 유지하면서, 어떤 state에서 최적의 action을 선택하는 역할
class Agent:
    def __init__(self, actions, epsilon=0.1, discount_factor=0.5, world_size=None):
        self.actions = actions
        self.values = defaultdict(float)
        self.epsilon = epsilon
        self.world_size = world_size
        self.discount_factor = discount_factor
        self.episode = list()

    def _possible_next_state(self, state):
        x, y = state
        next_state = [0.] * len(self.actions)
        # up, down, left, right

        if y == 0:
            next_state[0] = self.values[state]
        else:
            next_state[0] = self.values[(x, y - 1)]
        if y == self.world_size - 1:
            next_state[1] = self.values[state]
        else:
            next_state[1] = self.values[(x, y + 1)]
        if x == 0:
            next_state[2] = self.values[state]
        else:
            next_state[2] = self.values[(x - 1, y)]
        if x == self.world_size-1:
            next_state[3] = self.values[state]
        else:
            next_state[3] = self.values[(x + 1, y)]

        # print(next_state)
        return next_state

    def get_action(self, state):
        if np.random.uniform(0., 1.) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self._possible_next_state(state))

    def collect_sample(self, state, next_state, action, reward):
        if (state, action) not in [(s, a) for s, a, _, _ in self.episode]:
            self.episode.append((state, action, next_state, reward))

    def update(self):
        # im not sure....
        G = 0.
        for s, a, ns, r in reversed(self.episode):
            G = r + self.discount_factor * G
            value = self.values[ns]
            self.values[s] = value + self.discount_factor * (G - value)

    def reset(self):
        self.episode = list()

    def visualize_policy(self):
        world = np.zeros((self.world_size, self.world_size))
        for key in self.values.keys():
            x, y = key
            world[y, x] = self.values[key]

        fig = plt.figure()
        plt.imshow(world, cmap='jet')
        fig.show()

def main():
    world_size = 5
    n_actions = 4   # up, down, left, right
    env = Environment(world_size=world_size)
    agent = Agent(actions=np.arange(n_actions), world_size=world_size)

    for i in range(50000):
        state = env.reset()
        action = agent.get_action(state)

        while True:
            reward, next_state, is_term = env.step(state, action)
            agent.collect_sample(state, next_state, action, reward)

            if is_term:
                break
            state = next_state
            action = agent.get_action(state)
        agent.update()
        agent.reset()

    agent.visualize_policy()
    plt.show()

if __name__ == '__main__':
    main()