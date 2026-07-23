"""
session2_tictactoe.py -- vanilla ADVERSARIAL 2-player MCTS (UCT) on tic-tac-toe.

Reference: Survey
http://www.incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf

Algorithm has four stages (fig 2, pg 6):

  * UCB1 selection: Also used in session 1. Select action at each node by:
      argmax  W/N + c * sqrt( ln N_parent / N )
  * random moves for rollout till game ends (win, lose, draw)
  * PERSPECTIVE FLIPPING in backup - each node stores wins from the
    viewpoint of the player who MOVED INTO it, so a result of "X wins"
    adds +1 at nodes reached by X's moves and +0 at nodes reached by O's.
    Delete the flip (exercise!) and watch the agent start losing: it will
    happily walk into lines where its OPPONENT wins, because those look
    like "wins" to every node. This is the #1 MCTS implementation bug.

Verification targets (run this file):
  1. vs a random opponent, 200 games as X and as O: zero losses expected
  2. a "must block now" position: the chosen move blocks the threat
  3. self-play, both sides searching: tic-tac-toe is a draw, so every
     game should be drawn

Board: tuple of 9 chars in {'X', 'O', '.'}; X always moves first.
"""

from __future__ import annotations

import math
import random

#winning lines in terms of indices in a flattened array
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), #horizontals
         (0, 3, 6), (1, 4, 7), (2, 5, 8), #verticals
         (0, 4, 8), (2, 4, 6)] #diagonals

#empty board
EMPTY = tuple("." * 9)


def winner(board):
    for a, b, c in LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    return "draw" if "." not in board else None #None means game not ended yet


def to_move(board):
    return "X" if board.count("X") == board.count("O") else "O"


def play(board, cell):
    b = list(board)
    b[cell] = to_move(board) #don't need to explicitly keep player around
    return tuple(b)


def moves(board): #valid moves
    return [i for i, v in enumerate(board) if v == "."]


# ---------------------------------------------------------------------------
# Vanilla UCT.
# ---------------------------------------------------------------------------
class Node: #node in tree
    __slots__ = ("board", "parent", "move", "just_moved", "children",
                 "untried", "N", "W")

    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move

        # the player who made the move INTO this node ('.' for the root)
        self.just_moved = "." if parent is None else to_move(parent.board)

        self.children = [] #needs to be updated with each expansion move
        self.untried = moves(board) if winner(board) is None else []

        #node statistics - need to be updated with each move going through this node
        self.N = 0 #number of times board visited
        self.W = 0.0 #total win count from this point onwards


def result_for(player, w) -> float:
    """Value of outcome w from `player`'s perspective."""
    if w == "draw":
        return 0.5
    return 1.0 if w == player else 0.0


#core of mcts
def uct_search(board, iterations=1000, c=1.4, rng=None):
    rng = rng or random.Random()
    
    root = Node(board)
    #Stage A: explore/build tree in fixed compute budget
    for _ in range(iterations):
        node = root

        # 1. SELECT: descend fully expanded, non-terminal nodes via UCB1 (see session 1)
        while not node.untried and node.children: #to reiterate, the node has been visited and expanded
            log_n = math.log(node.N) #node.N = #times node was visited
            node = max(node.children, #expansion using UCB1 - first term = exploitation, second term = exploration
                       key=lambda ch: ch.W / ch.N + c * math.sqrt(log_n / ch.N))

        # 2. EXPAND: add one untried move (will eventually hit a new node/board state)
        if node.untried:
            m = rng.choice(node.untried) #pick a valid move randomly (uniform distribution)
            node.untried.remove(m)
            child = Node(play(node.board, m), parent=node, move=m) #current board, previous board, move that transitions
            node.children.append(child)
            node = child

        # 3. SIMULATE: random playout to the end
        b = node.board
        w = winner(b)
        while w is None:
            b = play(b, rng.choice(moves(b))) #note that we are randomly choosing from valid moves
            w = winner(b) #eventually will hit terminal state i.e. win/lose/draw

        # 4. BACKUP with perspective flip: each node scores the outcome
        #    from the viewpoint of the player who moved into it.
        while node is not None: #node here is the newly expanded node in line 113
            node.N += 1
            node.W += result_for(node.just_moved, w) #w=winner i..e 'X', 'O', 'draw'. for draw, W+=0.5 for both. for win/lose, only update for winner
            node = node.parent

    # Tree explored/built in #iterations. Now exploit knowledge
    # root = current board
    # root.children = set of valid moves in the current board
    # Play the most-visited child (robust child) -- the adversarial-game
    # convention, unlike the single-player "global best" of later sessions.
    best = max(root.children, key=lambda ch: ch.N)
    return best.move, root


def show(board):
    '''Printing
    '''
    rows = [" ".join(board[r * 3:r * 3 + 3]) for r in range(3)]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Verification.
# ---------------------------------------------------------------------------
def play_game(x_agent, o_agent, rng):
    b = EMPTY
    while winner(b) is None:
        agent = x_agent if to_move(b) == "X" else o_agent
        b = play(b, agent(b))
    return winner(b)


if __name__ == "__main__":
    rng = random.Random(0)
    # 1000 iterations plays perfectly here; at 400 the O side occasionally
    # drops a game to depth-4 double threats -- a nice "budget matters"
    # demo: lower this to 400 and rerun a few times.
    mcts_agent = lambda b: uct_search(b, iterations=1000, rng=rng)[0]
    rand_agent = lambda b: rng.choice(moves(b))

    # 1. never lose to a random opponent
    for side in "XO":
        w = l = d = 0
        for _ in range(200):
            agents = (mcts_agent, rand_agent) if side == "X" else (rand_agent, mcts_agent)
            r = play_game(*agents, rng)
            if r == "draw":
                d += 1
            elif r == side:
                w += 1
            else:
                l += 1
        print(f"MCTS as {side} vs random, 200 games:  won {w}  drew {d}  LOST {l}")

    # 2. must-block position: O threatens 6-7-8; X to move must take cell 8
    board = tuple("XX." + "..." + "OO.")
    #  X X .
    #  . . .
    #  O O .        <- X should WIN immediately at cell 2 (top row)...
    # careful: X also has a winning move! Good example: winning > blocking.
    move, _ = uct_search(board, iterations=800, rng=rng)
    print(f"\nposition:\n{show(board)}\nX plays cell {move} "
          f"(2 = take the win; anything else is a bug)")

    board = tuple("X.." + "..X" + "OO.")  # X=2, O=2: legal, X to move.
    # X . .
    # . . X
    # O O .   <- no X win available; O threatens cell 8: X must block.
    move, _ = uct_search(board, iterations=800, rng=rng)
    print(f"\nposition:\n{show(board)}\nX plays cell {move} "
          f"(8 = block; anything else is a bug)")

    # 3. self-play: perfect play draws tic-tac-toe
    strong = lambda b: uct_search(b, iterations=2000, rng=rng)[0]
    results = [play_game(strong, strong, rng) for _ in range(10)]
    print(f"\nself-play x10: {results.count('draw')}/10 draws "
          f"(all 10 expected)")

    # ----------------------------------------------------------------------
    # Exercises:
    #  * Break the flip: replace result_for(node.just_moved, w) with a
    #    constant perspective and re-run the tests. Instructive carnage.
    #  * Lower iterations to 20: which test fails first, and why?
    #  * Print the root's children (move, N, W/N) for the must-block
    #    position and watch visits concentrate on the blocking move.
    # ----------------------------------------------------------------------
