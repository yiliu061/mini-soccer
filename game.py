
import numpy as np
import random
import copy


np.random.seed(9999)

class Play_game:
    def __init__(self):
        self.world = World()

    def get_play_order(self):
        play_order = ['A', 'B']
        np.random.shuffle(play_order)

        return play_order

    def generate_all_states(self, grid):

        # grid: num_grids=2x4
        all_states = {}
        ball_pos = ['A', 'B']
        state_id = 0

        for ball in ball_pos:
            for pos_a in range(grid):
                for pos_b in range(grid):
                    if pos_a != pos_b:
                        all_states[ball + str(pos_a) + str(pos_b)] = state_id
                        state_id += 1

        return all_states

    def print_status(self, done, state_prime, rewards, all_states):

        print("State_label and index: {}, {}".format(state_prime, all_states[state_prime]))
        print("Rewards: {}".format(rewards))
        print("Goal: {}".format(done))
        print("-" * 30 + "\n")

    def init_game(self):

        num_rows = 2
        num_cols = 4
        grid = num_rows * num_cols

        self.play_order = self.get_play_order()
        self.all_states = self.generate_all_states(grid)

        init_state_A = Player(x=2, y=0, with_ball=False, player_name='A')
        init_state_B = Player(x=1, y=0, with_ball=True, player_name='B')

        self.world.place_player(init_state_A, player_name='A')
        self.world.place_player(init_state_B, player_name='B')
        self.world.set_goals(100, 0, 'A')
        self.world.set_goals(100, 3, 'B')

        init_state = self.world.get_state()
        if self.world.verbose:
            print("init_state:", init_state)
        # self.print_status(done = False, init_state = init_state, rewards={'A': 0, 'B': 0}, all_states=self.all_states)

        return self.all_states, init_state, self.play_order


    def play(self, actions):

        state_prime, rewards, done = self.world.move(actions, self.play_order)
        self.play_order = self.get_play_order()
        return state_prime, rewards, done, self.play_order


class Player:
    def __init__(self, x, y, with_ball, player_name=None):
        self.player_name = player_name
        self.x = x
        self.y = y
        self.with_ball = with_ball

    def update_state(self, x, y, with_ball):
        self.x = x
        self.y = y
        self.with_ball = with_ball

    def update_x(self, x):
        self.x = x

    def update_y(self, y):
        self.y = y

    def update_ball_pos(self, with_ball):
        self.with_ball = with_ball

class World:
    """ Soccer environment simulator class. """

    def __init__(self):
        """ Method that initializes the World class with class variables to be used in the class methods. """

        self.cols = 4
        self.rows = 2
        self.goal_r = {}
        self.players = {}
        self.goals = {}
        self.verbose = False


    def move(self, actions, play_order):

        new_pos = Player(0, 0, False)
        reward = 0.0
        done = False

        # play_order = self.get_play_order()

        for player_name in play_order:
            moving_player = self.players[player_name]
            opponent = set(play_order) - set(player_name)
            new_pos.update_state(moving_player.x, moving_player.y, moving_player.with_ball)
            # print "a[player_id]:", a, player_id, a[player_id]
            # action = self.actions[actions[player_id]]
            action = actions[player_name]

            if action == 0 and new_pos.y != 0:
                new_pos.update_y(new_pos.y - 1)

            elif action == 1 and new_pos.y != self.rows - 1:
                new_pos.update_y(new_pos.y + 1)

            elif action == 2 and new_pos.x < self.cols - 1:
                new_pos.update_x(new_pos.x + 1)

            elif action == 3 and new_pos.x > 0:
                new_pos.update_x(new_pos.x - 1)

            collision = self.check_collision(new_pos, moving_player, opponent)

            if not collision:
                moving_player.update_state(new_pos.x, new_pos.y, new_pos.with_ball)

            reward, done = self.check_goal(moving_player, play_order)

            if done:
                break

            reward, done = self.check_goal(self.players[list(opponent)[0]], play_order)

            if done:
                break
        state = self.get_state()

        if self.verbose:
            print('Player Order: {}'.format(play_order))
            print('Actions: {}'.format(actions))
            print('A location: ({}, {})'.format(self.players['A'].x, self.players['A'].y))
            print('B location: ({}, {})'.format(self.players['B'].x, self.players['B'].y))
            # print ""

        return state, reward, done


    def set_goals(self, goal_r, goal_col, player_name):

        self.goal_r[player_name] = goal_r
        self.goals[player_name] = goal_col


    def place_player(self, player, player_name):
        self.players[player_name] = copy.copy(player)


    def get_state(self):
        # e.g.[B12] is the init state
        state_label = self.players['A'].player_name if self.players['A'].with_ball else self.players['B'].player_name

        state_label += str(self.players['A'].y * self.cols + self.players['A'].x)
        state_label += str(self.players['B'].y * self.cols + self.players['B'].x)

        return state_label

    def get_players_states(self):
        return [self.players[p] for p in self.players]

    def plot_grid(self):
        grid = []
        for i in range(self.rows):
            # self.grid.append([' ','bg'] + ['  '] * (self.cols - 2) + ['ag', ' '])
            grid.append([' ','bg'] + ['  '] * (self.cols - 2) + ['ag', ' '])
        if len(self.players.keys()) > 0:

            for player_id in self.players:
                player = self.players[player_name]

                if player.with_ball:
                    cell = player_name + '*'

                else:
                    cell = player_name + ' '

                grid[player.y][player.x + 1] = cell

        for r in grid:
            print(' | '.join(r))

        print('')

    def check_collision(self, new_pos, moving_player, opponent):

        collision = False

        for op_name in opponent:
            oppo = self.players[op_name]

            if new_pos.x == oppo.x and new_pos.y == oppo.y:

                if self.verbose:
                    print('{} collided with {}'.format(moving_player.player_name, oppo.player_name))

                collision = True

                if new_pos.with_ball:
                    oppo.update_ball_pos(True)
                    moving_player.update_ball_pos(False)

                    if self.verbose:
                        print("{} steals from {}".format(oppo.player_name, moving_player.player_name))

        return collision

    def check_goal(self, moving_player, play_order):
        """ Method that verifies if a goal has been scored.

        Parameters
        ----------
            moving_player (Player): Player class instance
            self.play_order (list): List with the order in which each agent moves

        Returns
        -------
            tuple : 2-element tuple containing
                r (int): reward obtained from scoring a goal
                goal (bool): flag that indicates a goal has been scored

        """

        done = False
        opponent = set(play_order) - set(moving_player.player_name)
        r = {k: 0 for k in play_order}

        if moving_player.x == self.goals[moving_player.player_name] and moving_player.with_ball:

            if self.verbose:
                print("{} scored a goal!!".format(moving_player.player_name))

            done = True
            r[moving_player.player_name] = self.goal_r[moving_player.player_name]

            for op_name in opponent:
                r[op_name] = -self.goal_r[moving_player.player_name]

        else:
            other_goal = {op_name: moving_player.x == self.goals[op_name] and moving_player.with_ball
                          for op_name in play_order}

            if sum(other_goal.values()) > 0:

                if self.verbose:
                    print("{} scored an own goal!!".format(moving_player.player_name))

                done = True

                for k in other_goal.keys():

                    if other_goal[k]:
                        r[k] = self.goal_r[k]

                    else:
                        r[k] = -self.goal_r[k]

        return r, done





# def main():
#     game = Play_game()
#     game.init_game()
#     action = {'A': 1, 'B': 1}
#     state_prime, rewards, done = game.play(action)
#     action = {'A': 4, 'B': 3}
#     state_prime, rewards, done = game.play(action)
#
#
# if __name__ == '__main__':
#     main()
