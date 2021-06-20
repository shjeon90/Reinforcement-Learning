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

        if action == 0:     # up
            y -= 1
        elif action == 1:   # down
            y += 1
        elif action == 2:   # left
            x -= 1
        else:               # right
            x += 1

        if self._is_oob((x, y)):
            next_state = state
            return -2., next_state, False
        else:
            next_state = (x, y)
            if next_state in self.terms:
                return 0., next_state, True
            else: return -1., next_state, False


    def reset(self):
        while True:
            x, y = np.random.randint(self.world_size), np.random.randint(self.world_size)
            if (x, y) not in self.terms: break
        return x, y

class Agent:
    def __init__(self, actions, epsilon=0.1, learning_rate=0.1, discount_factor=0.5, world_size=None):
        self.actions_map = actions
        self.actions = np.arange(len(actions))
        self.values_1 = defaultdict(lambda : defaultdict(float))
        self.values_2 = defaultdict(lambda : defaultdict(float))
        self.episode = list()
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.world_size = world_size

    def get_action(self, state):
        if np.random.uniform(0., 1.) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            action_value_pair = [(a, self.values_1[state][a] + self.values_2[state][a]) for a in self.actions]
            if not action_value_pair:
                return np.random.choice(self.actions)
            action_value_pair.sort(key=lambda pair: -pair[1])
            return action_value_pair[0][0]


    def update_value_1(self, state, action, next_state, reward):
        value = self.values_1[state][action]
        action_value_pair = [(a, self.values_1[next_state][a]) for a in self.actions_map.keys()]
        action_value_pair.sort(key=lambda pair: -pair[1])
        value_ = self.values_2[next_state][action_value_pair[0][0]]
        self.values_1[state][action] = value + self.learning_rate * (reward + self.discount_factor * value_ - value)

    def update_value_2(self, state, action, next_state, reward):
        value = self.values_2[state][action]
        action_value_pair = [(a, self.values_2[next_state][a]) for a in self.actions_map.keys()]
        action_value_pair.sort(key=lambda pair: -pair[1])
        value_ = self.values_1[next_state][action_value_pair[0][0]]
        self.values_2[state][action] = value + self.learning_rate * (reward + self.discount_factor * value_ - value)

    def reset(self):
        # do nothing
        pass

    def print_policy_1(self):
        print('[*] print policy for Q_1')
        for y in range(self.world_size):
            for x in range(self.world_size):
                policy = [(a, self.values_1[(x, y)][a]) for a in self.values_1[(x, y)]]
                policy.sort(key=lambda pair: -pair[1])
                print('[-] state:', (x, y))
                for a, value in policy:
                    print('\t', self.actions_map[a], ':', value)
        print()

    def print_policy_2(self):
        print('[*] print policy for Q_2')
        for y in range(self.world_size):
            for x in range(self.world_size):
                policy = [(a, self.values_2[(x, y)][a]) for a in self.values_2[(x, y)]]
                policy.sort(key=lambda pair: -pair[1])
                print('[-] state:', (x, y))
                for a, value in policy:
                    print('\t', self.actions_map[a], ':', value)
        print()

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

        while True:
            action = agent.get_action(state)
            reward, next_state, is_term = env.step(state, action)

            if is_term:
                break

            r = np.random.uniform(0., 1.)
            if r < 0.5:
                agent.update_value_1(state, action, next_state, reward)
            else:
                agent.update_value_2(state, action, next_state, reward)
            state = next_state
        agent.reset()
    agent.print_policy_1()
    agent.print_policy_2()

if __name__ == '__main__':
    main()