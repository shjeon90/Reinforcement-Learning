import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from collections import deque


class Environment:
    def __init__(self, world_size):
        self.world_size = world_size
        self.terms = [(0, 0), (world_size - 1, world_size - 1)]

    def _is_oob(self, state):
        x, y = state
        if x < 0 or x >= self.world_size:
            return True
        elif y < 0 or y >= self.world_size:
            return True
        else:
            return False

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
            if (x, y) not in self.terms: break
        return x, y

class DqnAgent:
    def __init__(self, action_map, reply_mem_size, epsilon, gamma, n_batch, learning_rate):
        self.action_map = action_map
        self.actions = np.arange(len(action_map))
        self.n_actions = len(action_map)
        self.reply_mem = deque(maxlen=reply_mem_size)
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_batch = n_batch

        self.model = self.build()
        self.target = self.build()
        self.update_target_network()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate)

    def build(self):
        inputs = tf.keras.layers.Input([2])
        h = tf.keras.layers.Dense(50, activation='relu', kernel_initializer='he_uniform')(inputs)
        h = tf.keras.layers.Dense(50, activation='relu', kernel_initializer='he_uniform')(h)
        h = tf.keras.layers.Dense(50, activation='relu', kernel_initializer='he_uniform')(h)
        h = tf.keras.layers.Dense(self.n_actions, activation='linear', kernel_initializer='he_uniform')(h)

        model = tf.keras.models.Model(inputs, h)
        return model

    def get_action(self, state):
        state = np.array(state).reshape((1, -1))

        if np.random.uniform(0., 1.) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return tf.argmax(self.model(state)[0]).numpy()

    def sample_from_reply_mem(self, k):
        samples = random.sample(self.reply_mem, k)
        states = list()
        actions = list()
        rewards = list()
        next_states = list()
        is_terms = list()

        for sample in samples:
            s, a, r, ns, t = sample
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            is_terms.append(t)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        is_terms = np.array(is_terms)

        return states, actions, rewards, next_states, is_terms

    def update(self):
        if len(self.reply_mem) < self.n_batch:
            return

        tr_states, tr_actions, tr_rewards, tr_next_states, is_terms = \
            self.sample_from_reply_mem(self.n_batch)

        tr_y = list()
        for i in range(self.n_batch):
            if is_terms[i]:
                tr_y.append(tr_rewards[i])
            else:
                tmp = self.target(tr_next_states[i].reshape((1, -1)))[0]
                # print(tmp)
                tr_y.append(tr_rewards[i]
                            + self.gamma * tf.reduce_max(
                    tmp
                                            )
                )
        tr_y = tf.convert_to_tensor(tr_y)


        ind = np.concatenate((np.arange(self.n_batch).reshape((-1, 1))
                              , tr_actions.reshape((-1, 1))), 1)

        with tf.GradientTape() as g:
            print(tr_y)
            print(tf.gather_nd(self.model(tr_states, training=True), ind))
            print()
            loss = self.loss(tr_y, tf.gather_nd(self.model(tr_states, training=True), ind))
            # print(loss.numpy())
            var_list = self.model.trainable_variables
            grad = g.gradient(loss, var_list)
            self.opt.apply_gradients(zip(grad, var_list))
        return loss.numpy()

    def save_in_reply_mem(self, state, action, reward, next_state, is_term, world_size):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.reply_mem.append((state, action, reward, next_state, is_term))

    def update_target_network(self):
        self.target.set_weights(self.model.get_weights())

    def print_policy(self, world_size):
        x_space = np.arange(world_size)
        y_space = np.arange(world_size)
        X, Y = np.meshgrid(x_space, y_space)
        XY = np.dstack((X, Y)).reshape((-1, 2))

        Q = tf.reshape(self.model(XY), [world_size, world_size, self.n_actions])
        for y in range(world_size):
            for x in range(world_size):
                policy = np.argsort(-Q[y, x].numpy())
                print('[-] state:', (x, y), Q[y, x])
                for a in policy:
                    print('\t', self.action_map[a])



def main():
    world_size = 10
    reply_mem_size = 500
    epsilon = 1e-1
    n_episode = 400
    gamma = 9e-1
    n_batch = 3
    learning_rate = 1e-3
    action_map = {
        0: 'up', 1: 'down', 2: 'left', 3: 'right'
    }

    env = Environment(world_size=world_size)
    agent = DqnAgent(
        action_map=action_map,
        reply_mem_size=reply_mem_size,
        epsilon=epsilon,
        gamma=gamma,
        n_batch=n_batch,
        learning_rate=learning_rate)

    trajactories = list()
    for i in range(n_episode):
        trajactory = list()
        state = env.reset()
        trajactory.append(state)

        losses = list()
        for j in range(100):
            action = agent.get_action(state)
            reward, next_state, is_term = env.step(state, action)
            # print(state, action, next_state, reward, is_term)
            trajactory.append(next_state)

            agent.save_in_reply_mem(state, action, reward, next_state, is_term, world_size)
            loss = agent.update()
            if loss is not None: losses.append(loss)

            if is_term:
                break

            state = next_state
        print('[%d] avg. loss:' % i, np.mean(losses))
        agent.update_target_network()
        trajactories.append(trajactory)

    for t in trajactories:
        for s in t:
            print(s, '=>', end='')
        print()

    agent.print_policy(world_size)

if __name__ == '__main__':
    main()