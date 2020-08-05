#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-player Alpha Zero
@author: Thomas Moerland, Delft University of Technology
"""

import numpy as np
import tensorflow as tf
import argparse
import os
import time
import copy
# from gym import wrappers
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from helpers import (argmax, check_space, is_atari_game, copy_atari_state,
                     store_safely,
                     restore_atari_state, stable_normalizer, smooth,
                     symmetric_remove, Database)
from rl.make_game import make_game

plt.style.use('ggplot')

# Neural Networks #
class Model:

    def __init__(self, env, lr, n_hidden_layers, n_hidden_units):
        # Check the Gym environment
        self.action_dim, self.action_discrete = check_space(env.action_space)
        self.state_dim, self.state_discrete = check_space(env.observation_space)
        if not self.action_discrete:
            raise ValueError('Continuous action space not implemented')

        # Placeholders
        # if not self.state_discrete:
        #     self.x = x = tf.placeholder("float32",
        #                                 shape=np.append(None, self.state_dim),
        #                                 name='x')  # state
        # else:
        #     self.x = x = tf.placeholder("int32",
        #                                 shape=np.append(None, 1))  # state
        #     x = tf.squeeze(tf.one_hot(x, self.state_dim, axis=1), axis=2)
        #
        # # Feedforward: Can be modified to any representation function,
        # e.g. convolutions, residual networks, etc.
        # for i in range(n_hidden_layers):
        #     x = slim.fully_connected(x, n_hidden_units,
        #     activation_fn=tf.nn.elu)

        # Remy

        # If discrete
        # keras_input = keras.Input(shape=(1,), dtype='int32')
        # x = tf.one_hot(indices=keras_input, depth=10)
        # x = tf.squeeze(x, axis=[1])

        # If continuous
        keras_input = keras.Input(shape=self.state_dim, dtype='float32')
        x = keras_input

        for i in range(n_hidden_layers):
            x = layers.Dense(n_hidden_units)(x)
            x = layers.Activation('relu')(x)

        policy_head = layers.Dense(self.action_dim, activation='softmax',
                                   name='policy')(x)
        self.pi_hat = policy_head

        value_head = layers.Dense(1, name='value')(x)
        self.v_hat = value_head

        self.model = keras.Model(inputs=keras_input,
                                 outputs=[policy_head, value_head])

        optimizer = tf.optimizers.Adam(learning_rate=lr)

        self.model.compile(
                optimizer=optimizer,
                loss={'value': 'mse',
                      'policy': 'categorical_crossentropy'},
                metrics=['acc'])

        # rEMY

        # Output
        # log_pi_hat = slim.fully_connected(x, self.action_dim,
        #                                   activation_fn=None)
        # self.pi_hat = tf.nn.softmax(log_pi_hat)  # policy head
        # self.v_hat = slim.fully_connected(x, 1,
        #                                   activation_fn=None)  # value head
        #
        # # Loss
        # self.V = tf.placeholder("float32", shape=[None, 1], name='V')
        # self.pi = tf.placeholder("float32", shape=[None, self.action_dim],
        #                          name='pi')
        # self.V_loss = tf.losses.mean_squared_error(labels=self.V,
        #                                            predictions=self.v_hat)
        # self.pi_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        #         labels=self.pi, logits=log_pi_hat)
        # self.loss = self.V_loss + tf.reduce_mean(self.pi_loss)
        #
        # self.lr = tf.Variable(lr, name="learning_rate", trainable=False)
        # optimizer = tf.train.rMSPropOptimizer(learning_rate=lr)
        # self.train_op = optimizer.minimize(self.loss)

    def train(self, sb, v_batch, pi_batch):
        # self.sess.run(self.train_op, feed_dict={self.x: sb,
        #                                         self.v: v_batch,
        #                                         self.pi: pi_batch})
        return self.model.train_on_batch(x=sb, y=[pi_batch, v_batch])

    def predict_V(self, s):

        return self.model.predict(s)[1]

        # return self.sess.run(self.v_hat, feed_dict={self.x: s})

    def predict_pi(self, s):

        return self.model.predict(s)[0]
        # return self.sess.run(self.pi_hat, feed_dict={self.x: s})


# MCTS functions #

class Action:
    """ Action object """

    def __init__(self, index, parent_state, q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.w = 0.0
        self.n = 0
        self.q = q_init
        # self.child_state = None

    def add_child_state(self, s1, r, terminal, model):
        self.child_state = State(s1, r, terminal, self, self.parent_state.na,
                                 model)
        return self.child_state

    def update(self, r):
        self.n += 1
        self.w += r
        self.q = self.w / self.n


class State:
    """ State object """

    def __init__(self, index, r, terminal, parent_action, na, model):
        """ Initialize a new state """
        self.index = index  # state
        self.r = r  # reward upon arriving in this state
        self.terminal = terminal  # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model
        # Remy init like that
        self.v = 0
        # Fin Remy
        self.evaluate()
        # Child actions
        self.na = na
        self.child_actions = [Action(a, parent_state=self, q_init=self.v) for a
                              in range(na)]
        self.priors = model.predict_pi(index[None,]).flatten()

    def select(self, c=1.5):
        """ Select one of the child actions based on UCT rule """
        uct = np.array([child_action.q + prior * c * (
                np.sqrt(self.n + 1) / (child_action.n + 1)) for
                        child_action, prior in
                        zip(self.child_actions, self.priors)])
        winner = argmax(uct)
        return self.child_actions[winner]

    def evaluate(self):
        """ Bootstrap the state value """
        self.v = np.squeeze(self.model.predict_V(
                self.index[None,])) if not self.terminal else np.array(0.0)

    def update(self):
        """ update count on backward pass """
        self.n += 1


class MCTS:
    """ MCTS object """

    def __init__(self, root_index, model, na, gamma):
        self.root = None
        self.root_index = root_index
        self.model = model
        self.na = na
        self.gamma = gamma

    def search(self, n_mcts, c, env, mcts_env):
        """ Perform the MCTS search from the root """
        if self.root is None:
            self.root = State(self.root_index, r=0.0, terminal=False,
                              parent_action=None, na=self.na,
                              model=self.model)  # initialize new root
        else:
            self.root.parent_action = None  # continue from current root
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(env)
        if is_atari:
            snapshot = copy_atari_state(
                    env)  # for Atari: snapshot the root at the beginning
            restore_atari_state(mcts_env, snapshot)

        for i in range(n_mcts):
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(
                        env)  # copy original env to rollout from

            while not state.terminal:
                action = state.select(c=c)
                s1, r, t, _ = mcts_env.step(action.index)
                if hasattr(action, 'child_state'):
                    state = action.child_state  # select
                    continue
                else:
                    state = action.add_child_state(s1, r, t,
                                                   self.model)  # expand
                    break

            # Back-up
            r = state.v
            # loop back-up until root is reached
            while state.parent_action is not None:
                r = state.r + self.gamma * r
                action = state.parent_action
                action.update(r)
                state = action.parent_state
                state.update()

    def return_results(self, temp):
        """ Process the output at the root node """
        counts = np.array(
                [child_action.n for child_action in self.root.child_actions])
        q = np.array(
                [child_action.q for child_action in self.root.child_actions])
        pi_target = stable_normalizer(counts, temp)
        v_target = np.sum((counts / np.sum(counts)) * q)[None]
        return self.root.index, pi_target, v_target

    def forward(self, a, s1):
        """ Move the root forward """
        if not hasattr(self.root.child_actions[a], 'child_state'):
            self.root = None
            self.root_index = s1
        elif np.linalg.norm(
                self.root.child_actions[a].child_state.index - s1) > 0.01:
            print(
                    'warning: this domain seems stochastic. Not re-using the '
                    'subtree for next search. ' +
                    'To deal with stochastic environments, implement '
                    'progressive widening.')
            self.root = None
            self.root_index = s1
        else:
            self.root = self.root.child_actions[a].child_state


# Agent #
def agent(game, n_ep, n_mcts, max_ep_len, lr, c, gamma, data_size, batch_size,
          temp, n_hidden_layers, n_hidden_units):
    """ Outer training loop """
    seed_best = None
    a_best = None
    episode_returns = []  # storage
    timepoints = []
    # environments
    env = make_game(game)
    is_atari = is_atari_game(env)
    mcts_env = make_game(game) if is_atari else None

    database = Database(max_size=data_size, batch_size=batch_size)
    model = Model(env=env, lr=lr, n_hidden_layers=n_hidden_layers,
                  n_hidden_units=n_hidden_units)
    t_total = 0  # total steps
    r_best = -np.Inf

    for ep in range(n_ep):
        start = time.time()
        s = env.reset()
        r2 = 0.0  # Total return counter
        a_store = []
        seed = np.random.randint(1e7)  # draw some env seed
        env.seed(seed)
        if is_atari:
            mcts_env.reset()
            mcts_env.seed(seed)

        mcts = MCTS(root_index=s, model=model,
                    na=model.action_dim,
                    gamma=gamma)  # the object responsible for MCTS searches
        for t in range(max_ep_len):
            # MCTS step
            mcts.search(n_mcts=n_mcts, c=c, env=env,
                        mcts_env=mcts_env)  # perform a forward search
            state, pi, v = mcts.return_results(
                    temp)  # extract the root output
            database.store((state, v, pi))

            # Make the true step
            a = np.random.choice(len(pi), p=pi)
            a_store.append(a)
            s1, r, terminal, _ = env.step(a)
            r2 += r
            # total number of environment steps (counts the mcts steps)
            t_total += n_mcts

            if terminal:
                break
            else:
                mcts.forward(a, s1)

        # Finished episode
        episode_returns.append(r2)  # store the total episode return
        timepoints.append(
                t_total)  # store the timestep count of the episode return
        store_safely(os.getcwd(), 'result',
                     {'r': episode_returns, 't': timepoints})

        if r2 > r_best:
            a_best = a_store
            seed_best = seed
            r_best = r2
        print(
            'Finished episode {}, total return: {}, total time: {} sec'.format(
                        ep, np.round(r2, 2),
                        np.round((time.time() - start), 1)))
        # Train
        database.reshuffle()
        for epoch in range(1):
            for sb, v_batch, pi_batch in database:
                model.train(sb, v_batch, pi_batch)
    # return results
    return episode_returns, timepoints, a_best, seed_best, r_best


# Command line call, parsing and plotting #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='CartPole-v0',
                        help='Training environment')
    parser.add_argument('--n_ep', type=int, default=50,
                        help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=25,
                        help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=150,
                        help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--temp', type=float, default=21.0,
                        help='Temperature in normalization of counts to policy')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount parameter')
    parser.add_argument('--data_size', type=int, default=1000,
                        help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Minibatch size')
    parser.add_argument('--window', type=int, default=10,
                        help='Smoothing window for visualization')

    parser.add_argument('--n_hidden_layers', type=int, default=2,
                        help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=128,
                        help='Number of units per hidden layers in NN')

    args = parser.parse_args()
    episode_returns, timepoints, a_best, seed_best, r_best = agent(
            game=args.game, n_ep=args.n_ep, n_mcts=args.n_mcts,
            max_ep_len=args.max_ep_len, lr=args.lr, c=args.c, gamma=args.gamma,
            data_size=args.data_size, batch_size=args.batch_size,
            temp=args.temp,
            n_hidden_layers=args.n_hidden_layers,
            n_hidden_units=args.n_hidden_units)

    # Finished training: Visualize
    fig, ax = plt.subplots(1, figsize=[7, 5])
    total_eps = len(episode_returns)
    episode_returns = smooth(episode_returns, args.window, mode='valid')
    ax.plot(symmetric_remove(np.arange(total_eps), args.window - 1),
            episode_returns, linewidth=4, color='darkred')
    ax.set_ylabel('return')
    ax.set_xlabel('Episode', color='darkred')
    plt.savefig(os.getcwd() + '/learning_curve.png', bbox_inches="tight",
                dpi=300)

#    print('Showing best episode with return {}'.format(r_best))
#    env = make_game(args.game)
#    env = wrappers.Monitor(env,os.getcwd() + '/best_episode',force=True)
#    env.reset()
#    env.seed(seed_best)
#    for a in a_best:
#        env.step(a)
#        env.render()
