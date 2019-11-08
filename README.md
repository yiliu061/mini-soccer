# Mini-soccer

Implementation of multiagent Correlated-Q(CE-Q), Foe-Q, Friend-Q, and regular Q-learning for playing a mini version of soccer game.

## Environment

The multi-step, zero-sum, grid-based soccer game is described in Greenwald and Hall’s 2003 paper<sup>\[1\]</sup>.

The soccer field in this game is a 2 x 4 grid. The game starts with player A and B in state *s* where B has the ball and players move in random order. Both palyer choose from five action space, namely N, S, E, W, and stick simultaneously. If the player with the ball moves into his goal, he scores +100 and the other player gets -100. On the other hand, if he moves to the oppenent’s goal, he gets -100 and the opponent gets +100. In either case, the game ends. If this sequence of actions causes the players to collide, only the first player moves. The ball changes possession when the palyer without the ball moves first to where the one with the ball is.

![](graphs/env.png)
