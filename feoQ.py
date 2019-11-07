
import numpy as np
import random as rand
from itertools import *
from collections import deque
from copy import deepcopy
from scipy.optimize import linprog
import matplotlib.pyplot as plt

from game import Player, World, Play_game

rand.seed(9999)


class FoeQLearner:
    def __init__(self,
                 num_states=(2 * 4) * (2 * 4) * 2,
                 num_actions=5,
                 alpha=1.0,
                 alpha_decay=0.99997,
                 gamma=0.9,
                 epsilon=1.0,
                 max_steps_per_episode=1000,
                 max_num_episodes=1000000,
                 save_model_per=20000,
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



    def foeQ_agent(self):

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
                self.update_Q(current_actions, state_prime, r['A'], 'A')
                self.update_Q(current_actions, state_prime, r['B'], 'B')
                self.state = state_prime

                self.step_count += 1

                # save and plot
                if self.step_count > 0 and self.step_count % self.save_model_per == 0:
                    experiment_id = 3
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

        # if t == 0:
        #     self.actions = actions

        return actions


    def get_V(self, q_table, state_prime):

        num_states, num_action_0, num_action_1 = q_table.shape

        # -1.0 because we need to maximize
        c = [-1.0] + [0.0] * num_action_0

        A_ub = np.transpose(np.concatenate([[[1.0] * num_action_0], -q_table[state_prime]], 0))
        b_ub = [0.0] * num_action_1

        A_eq = [[0.0] + [1.0] * num_action_0]
        b_eq = [1.0]

        bounds = [[None, None]] + [[0.0, None]] * num_action_0

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        V = res.x[0] if type(res.x) != float else 1.0
        # V = res.x[0]
        return V



    def update_Q(self, actions, state_prime, r, player_name):

        state_index = self.all_states[self.state]
        state_prime_index = self.all_states[state_prime]

        # update Q table
        # V = np.amax(self.q_tables[player_name][state_prime_index])
        V = self.get_V(self.q_tables[player_name], state_prime_index)

        error = (((1 - self.gamma) * r + self.gamma * V) - self.q_tables[player_name][state_index, actions['A'], actions['B']]) * self.alpha

        self.q_tables[player_name][state_index, actions['A'], actions['B']] += error


        # collect errors
        if player_name == 'A' and self.state == 'B21' and actions['A'] == 1 and actions['B'] == 4:
            self.ERRs.append(abs(error))
            self.steps_to_plot.append(self.step_count)
            # print self.ERRs

        # if self.verbose:
        #     print 'Action of B at state s: ', np.argmax(self.q_tables['B'][self.all_states['B21']]) % 5


    def plot_error(self, experiment_id):


        err_to_plot = self.ERRs
        step_to_plot = self.steps_to_plot


        plt.plot(step_to_plot, err_to_plot, '-', linewidth=0.5)
        plt.ylim(0, 0.5)
        plt.xlim(0, 1000000)
        plt.xlabel("Simulation Iteration")
        plt.ylabel("Q-value Difference")
        plt.title("Foe-Q")
        plt.savefig('outputs/FoeQ_exp_' + str(experiment_id) + '_' + str(self.step_count) + '.png')
        plt.show(block=False)
        plt.show()


    def save_data(self, experiment_id):
        error_file_name = 'outputs/data_FeoQ_error_exp_' + str(experiment_id) + '.txt'
        error_file = open(error_file_name, 'w')
        for item in self.ERRs:
            error_file.write("%s\n" % item)

        # step_file = open('step_test.txt', 'w')
        step_file_name = 'outputs/data_FoeQ_step_exp_' + str(experiment_id) + '.txt'
        step_file = open(step_file_name, 'w')
        for item in self.steps_to_plot:
            step_file.write("%s\n" % item)

        if self.verbose:
            print(self.ERRs)
            print('epsilon:', self.epsilon)
            print('alpha: ', self.alpha)





def main():
    learner = FoeQLearner()
    learner.foeQ_agent()


if __name__ == '__main__':
    main()




