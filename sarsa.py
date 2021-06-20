import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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
        # x, y = state
        #
        # if action == 0:  # up
        #     y = y if y == 0 else y - 1
        # elif action == 1:  # down
        #     y = y if y == self.world_size - 1 else y + 1
        # elif action == 2:  # left
        #     x = x if x == 0 else x - 1
        # else:  # right
        #     x = x if x == self.world_size - 1 else x + 1
        #
        # next_state = (x, y)
        #
        # if (x, y) in self.terms:
        #     return 0., next_state, True
        # else:
        #     return -1., next_state, False

    def reset(self):
        while True:
            x, y = np.random.randint(self.world_size), np.random.randint(self.world_size)
            if (x, y) not in self.terms: break
        return x, y

class Agent:
    def __init__(self, actions, epsilon=0.1, learning_rate=0.1, discount_factor=0.5, world_size=None):
        self.actions_map = actions
        self.actions = np.arange(len(actions))
        # self.values = defaultdict(float)
        self.values = defaultdict(lambda : defaultdict(float))
        self.episode = list()
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.world_size = world_size

    def get_action(self, state):
        if np.random.uniform(0., 1.) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            action_value_pair = [(a, self.values[state][a]) for a in self.actions]
            if not action_value_pair:
                return np.random.choice(self.actions)
            action_value_pair.sort(key=lambda pair:-pair[1])
            return action_value_pair[0][0]


    def update(self, state, action, next_state, next_action, reward):
        value = self.values[state][action]
        value_ = self.values[next_state][next_action]

        self.values[state][action] = value + self.learning_rate * (reward + self.discount_factor * value_ - value)

    def reset(self):
        # do nothing
        pass

    def print_policy(self):
        for y in range(self.world_size):
            for x in range(self.world_size):
                # policy = [(s, a, self.values[(s, a)]) for s, a in self.values.keys() if s == (x, y)]

                policy = [(a, self.values[(x, y)][a]) for a in self.values[(x, y)]]
                policy.sort(key=lambda pair: -pair[1])
                print('[-] state:', (x, y))
                for a, value in policy:
                    print('\t', self.actions_map[a], ':', value)


def main():
    world_size = 5
    n_actions = 4
    learning_rate = 0.1
    epsilon = 0.1
    discount_factor = 0.5
    actions = {
        0: 'up', 1: 'down', 2: 'left', 3: 'right'
    }
    env = Environment(world_size=world_size)
    agent = Agent(actions=actions, epsilon=epsilon, learning_rate=learning_rate, discount_factor=discount_factor, world_size=world_size)

    for i in range(10000):
        state = env.reset()
        action = agent.get_action(state)

        while True:
            reward, next_state, is_term = env.step(state, action)
            next_action = agent.get_action(next_state)

            if is_term:
                break
            agent.update(state, action, next_state, next_action, reward)
            state = next_state
            action = next_action
        agent.reset()
    agent.print_policy()

if __name__ == '__main__':
    main()