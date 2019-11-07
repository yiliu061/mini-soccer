
import numpy as np
import random as rand
from itertools import *
from collections import deque
from copy import deepcopy
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import os.path

from game import Player, World, Play_game

rand.seed(9999)


class CeQLearner:
    def __init__(self,
                 num_states=(2 * 4) * (2 * 4) * 2,
                 num_actions=5,
                 alpha=1.0,
                 alpha_decay=0.99997,
                 gamma=0.9,
                 epsilon=1.0,
                 max_steps_per_episode=1000, #1000
                 max_num_episodes=10000000, #10000000
                 save_model_per=10000,
                 verbose=True):

        self.world = World()
        self.game = Play_game()

        # inputs
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_steps_per_episode = max_steps_per_episode
        self.max_num_episodes = max_num_episodes
        self.save_model_per = save_model_per

        # initialize
        self.num_states = num_states  # 4x2 grid, two players, with or without ball
        self.num_actions = num_actions

        q_table = np.full(shape=(num_states, num_actions, num_actions), fill_value=1.0)
        self.q_tables = {'A': deepcopy(q_table), 'B': deepcopy(q_table)}

        self.state = {}

        # error
        self.ERRs = []
        self.steps_to_plot = []

        self.verbose = verbose



    def ceQ_agent(self):

        self.step_count = 1

        for episode in range(self.max_num_episodes):
            # reset game
            self.all_states, self.state, _ = self.game.init_game()

            # play game and learn
            for t in range(self.max_steps_per_episode):

                # get actions
                current_actions = self.get_actions()

                # observe
                state_prime, r, done, _ = self.game.play(current_actions)

                # update
                self.update_Q(current_actions, state_prime, r['A'], r['B'])
                self.state = state_prime

                self.step_count += 1

                # save and plot
                if self.step_count > 0 and self.step_count % self.save_model_per == 0:

                    experiment_id = 1
                    self.save_data(experiment_id)
                    self.plot_error(experiment_id)

                if done:
                    self.alpha *= self.alpha_decay
                    break

            if self.step_count > 1000000:
                break


    def get_actions(self):
        # off-policy
        actions = {}
        actions['A'] = rand.randint(0, self.num_actions - 1)
        actions['B'] = rand.randint(0, self.num_actions - 1)

        return actions


    def get_V(self, q_table_0, q_table_1, state_prime):
        num_states, num_action_0, num_action_1 = q_table_0.shape

        q_table_0 = q_table_0[state_prime]
        q_table_1 = q_table_1[state_prime]

        c = - (q_table_0.flatten() + q_table_1.flatten())

        A_ub = []

        for i in range(num_action_0):
          for j in range(num_action_0):
            if i != j:
              A_ub.append(np.vstack([
                np.zeros((i, num_action_1)),
                q_table_0[j, :] - q_table_0[i, :],
                np.zeros((num_action_0 - i - 1, num_action_1))]).flatten())

        for i in range(num_action_1):
          for j in range(num_action_1):
            if i != j:
              A_ub.append(np.vstack([
                np.zeros((i, num_action_0)),
                q_table_1[:, j] - q_table_1[:, i],
                np.zeros((num_action_1 - i - 1, num_action_0))]).transpose().flatten())

        A_ub = np.stack(A_ub, 0)
        b_ub = np.zeros([A_ub.shape[0]])

        A_eq = [[1.0] * num_action_0 * num_action_1]
        b_eq = [1.0]

        bounds = [[0.0, None]] * num_action_0 * num_action_1

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        return res


    def update_Q(self, actions, state_prime, r_A, r_B):

        state_index = self.all_states[self.state]
        state_prime_index = self.all_states[state_prime]

        # update Q table
        # V = np.amax(self.q_tables[player_name][state_prime_index])
        res = self.get_V(self.q_tables['A'], self.q_tables['B'], state_prime_index)

        if res.success:
          V = - res.fun
          error_A = (((1 - self.gamma) * r_A + self.gamma * V) - self.q_tables['A'][state_index, actions['A'], actions['B']]) * self.alpha
          error_B = (((1 - self.gamma) * r_B + self.gamma * V) - self.q_tables['B'][state_index, actions['A'], actions['B']]) * self.alpha
        else:
          error_A = 0.
          error_B = 0.

        self.q_tables['A'][state_index, actions['A'], actions['B']] += error_A
        self.q_tables['B'][state_index, actions['A'], actions['B']] += error_B

        # collect errors
        if self.state == 'B21' and actions['A'] == 1 and actions['B'] == 4:
            self.ERRs.append(abs(error_A))
            self.steps_to_plot.append(self.step_count)
            # print self.ERRs

        # if self.verbose:
        #     print 'Action of B at state s: ', np.argmax(self.q_tables['B'][self.all_states['B21']]) % 5


    def plot_error(self, experiment_id):

        err_to_plot = self.ERRs
        step_to_plot = self.steps_to_plot


        plt.plot(step_to_plot, err_to_plot, '-', linewidth=0.8)
        plt.ylim(0, 0.5)
        # plt.xlim(0, 1000000)
        plt.xlabel("Simulation Iteration")
        plt.ylabel("Q-value Difference")
        plt.title("Ce-Q")
        plt.savefig('outputs/CeQ_exp_' + str(experiment_id) + '_' + str(self.step_count) + '.png')

        plt.show(block=False)
        plt.show()


    def save_data(self, experiment_id):
        error_file_name = 'outputs/data_CeQ_error_exp_' + str(experiment_id) + '.txt'
        error_file = open(error_file_name, 'w')
        for item in self.ERRs:
            error_file.write("%s\n" % item)

        step_file_name = 'outputs/data_CeQ_step_exp_' + str(experiment_id) + '.txt'
        step_file = open(step_file_name, 'w')
        for item in self.steps_to_plot:
            step_file.write("%s\n" % item)

        if self.verbose:
            print(self.ERRs)
            print('epsilon:', self.epsilon)
            print('alpha: ', self.alpha)





def main():
    learner = CeQLearner()
    learner.ceQ_agent()


if __name__ == '__main__':
    main()




