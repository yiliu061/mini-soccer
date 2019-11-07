
import numpy as np
import random as rand
from itertools import *
from collections import deque
from copy import deepcopy
import matplotlib.pyplot as plt

from game import Player, World, Play_game

rand.seed(9999)


class QLearner:
    def __init__(self,
                 num_states=(2 * 4) * (2 * 4) * 2,
                 num_actions=5,
                 alpha=1.0,
                 alpha_decay=0.99996,
                 gamma=0.9,
                 epsilon=1.0,
                 epsilon_decay=0.99996,
                 max_steps_per_episode=1000,
                 max_num_episodes=1000000,
                 save_model_per=1000000,
                 verbose=False):

        self.world = World()
        self.game = Play_game()

        # inputs
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.max_steps_per_episode = max_steps_per_episode
        self.max_num_episodes = max_num_episodes
        self.save_model_per = save_model_per

        # initialize
        self.num_states = num_states  # 4x2 grid, two players, with or without ball
        self.num_actions = num_actions

        self.q_table = np.full(shape=(num_states, num_actions), fill_value=1.0)
        self.q_tables = {'A': deepcopy(self.q_table), 'B': deepcopy(self.q_table)}

        self.state = {}
        self.actions = {'A': 0, 'B': 0}  # map N, S, E, W, and stick to [0,1,2,3,4]

        # error
        self.ERRs = []
        self.steps_to_plot = []

        self.verbose = verbose

    def Qlearning_agent(self, verbose):

        self.step_count = 1

        for episode in range(self.max_num_episodes):
            # reset game
            self.all_states, self.state, _ = self.game.init_game()
            self.actions = self.get_first_actions(self.state)

            # play game and learn
            for t in range(self.max_steps_per_episode):
                # update q_table seperately
                state_prime, r, done, _ = self.game.play(self.actions)
                self.actions['A'] = self.update_Q(state_prime, r['A'], 'A')
                self.actions['B'] = self.update_Q(state_prime, r['B'], 'B')

                self.state = state_prime
                self.step_count += 1

                # save and plot
                if self.step_count > 0 and self.step_count % self.save_model_per == 0:
                    experiment_id = 1
                    self.save_data(experiment_id)
                    self.plot_error(experiment_id)

                if done:
                    self.epsilon *= self.epsilon_decay
                    self.alpha *= self.alpha_decay
                    break

            if self.step_count > 1000000:
                break


    def get_first_actions(self, s):

        first_actions = {}

        first_actions['A'] = rand.randint(0, self.num_actions - 1)
        first_actions['B'] = rand.randint(0, self.num_actions - 1)

        if self.verbose: print('s =', s, 'a =', first_actions)

        return first_actions

    def update_Q(self, state_prime, r, player_name):

        # epsilon-greedy selection function
        state_index = self.all_states[self.state]
        state_prime_index = self.all_states[state_prime]

        if rand.random() < self.epsilon:
            action = rand.randint(0, self.num_actions - 1)

        else:
            action = np.argmax(self.q_tables[player_name][state_prime_index])

        # update Q table
        V = np.amax(self.q_tables[player_name][state_prime_index])

        error = (((1 - self.gamma) * r + self.gamma * V) - self.q_tables[player_name][
            state_index, self.actions[player_name]]) * self.alpha

        self.q_tables[player_name][state_index, self.actions[player_name]] += error

        # print self.q_tables[player_name]

        # collect errors
        if player_name == 'A' and self.state == 'B21' and self.actions['A'] == 1 and error != 0.0:
            self.ERRs.append(abs(error))
            self.steps_to_plot.append(self.step_count)
            # print self.ERRs

        if self.verbose: print('s =', state_prime, 'a =', action, 'r =', r)
        return action

    def plot_error(self, experiment_id):

        err_to_plot = self.ERRs[::20]
        step_to_plot = self.steps_to_plot[::20]

        plt.plot(step_to_plot, err_to_plot, '-', linewidth=0.3)
        plt.ylim(0, 0.5)
        plt.xlim(0, 1000000)
        plt.xlabel("Simulation Iteration")
        plt.ylabel("Q-value Difference")
        plt.title("Q-learner")

        plt.savefig('outputs/Q_exp_' + str(experiment_id) + '_' + str(self.step_count) + '.png')
        plt.show(block=False)
        plt.show()


    def save_data(self, experiment_id):
        error_file_name = 'outputs/data_Q_error_exp_' + str(experiment_id) + '.txt'
        error_file = open(error_file_name, 'w')
        for item in self.ERRs:
            error_file.write("%s\n" % item)

        # step_file = open('step_test.txt', 'w')
        step_file_name = 'outputs/data_Q_step_exp_' + str(experiment_id) + '.txt'
        step_file = open(step_file_name, 'w')
        for item in self.steps_to_plot:
            step_file.write("%s\n" % item)

        if self.verbose:
            print(self.ERRs)
            print('epsilon:', self.epsilon)
            print('alpha: ', self.alpha)





def main():
    qlearner = QLearner()
    qlearner.Qlearning_agent()


if __name__ == '__main__':
    main()




