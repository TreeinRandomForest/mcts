"""
mcts_core.py -- a minimal single-player MCTS engine for teaching.

This is the engine that stays fixed across tutorial sessions; only the
"game" object plugged into it changes. It implements the four classic
phases (selection / expansion / evaluation / backup) with the
single-player modifications discussed in the tutorial:

  * No opponent, so NO perspective flipping during backup.
  * Backup tracks both the MEAN and the MAX of values seen below each
    node; selection uses a blend  Q = alpha * max + (1 - alpha) * mean.
    (alpha = 0 recovers vanilla UCT-style averaging; alpha -> 1 turns
    the search into best-first hunting for a single great line.)
  * Selection uses PUCT:  score = Q + c_puct * P(a) * sqrt(N_parent) / (1 + N_child)
    where P(a) is a prior supplied by the game (uniform by default).
  * The final answer is the GLOBAL BEST state ever evaluated (the
    "incumbent"), not the most-visited root child.
  * A transition may fail (return None): the child becomes a terminal
    node with a fixed penalty value. Failures are information.

The `evaluate(state)` method is deliberately the only source of value:
  - In classic MCTS it is a random ROLLOUT from the state (an estimate).
  - In "AlphaZero-style" setups it is a value-network call.
  - In ground-truth setups (session 6: compile-and-benchmark) it is a
    real measurement.
The engine does not care which -- "simulation" is just one possible
implementation of an evaluation function.
"""

from __future__ import annotations

import math
import random


# --------------------------------------------------------------------------
# The game interface: the ONLY thing you swap between domains.
# --------------------------------------------------------------------------
class SinglePlayerGame:
    """Subclass this and implement the four core methods."""

    # If True, evaluate() returns ground truth for the state itself, so the
    # engine updates its incumbent on EVERY evaluation (session 6).
    # If False, evaluate() is only an estimate (e.g. a random rollout), so
    # the incumbent is only updated at terminal states, and rollout-based
    # games should track their own best-seen terminal internally (session 3).
    evaluation_is_ground_truth = False

    def legal_actions(self, state) -> list:
        raise NotImplementedError

    def transition(self, state, action):
        """Return the successor state, or None if the action failed
        (e.g. generated code that does not compile)."""
        raise NotImplementedError

    def is_terminal(self, state) -> bool:
        raise NotImplementedError

    def evaluate(self, state) -> float:
        """Value estimate (rollout) or ground-truth reward for `state`."""
        raise NotImplementedError

    # Optional: action priors P(s, a). Uniform by default. Session 4
    # replaces this with a heuristic; Track 1 replaces it with LLM
    # relevance scores masked by static-analysis feasibility.
    def priors(self, state, actions) -> dict:
        p = 1.0 / max(len(actions), 1)
        return {a: p for a in actions}


# --------------------------------------------------------------------------
# Tree node.
# --------------------------------------------------------------------------
class Node:
    __slots__ = ("state", "parent", "action", "prior", "children",
                 "untried", "action_priors", "N", "W", "Qmax",
                 "value", "terminal")

    def __init__(self, state, parent=None, action=None, prior=1.0,
                 terminal=False):
        self.state = state          # None for failed transitions
        self.parent = parent
        self.action = action        # action that led here (edge label)
        self.prior = prior          # P(parent_state, action)
        self.children = {}          # action -> Node
        self.untried = None         # actions not yet expanded (lazy)
        self.action_priors = None
        self.N = 0                  # visit count
        self.W = 0.0                # sum of backed-up values (for the mean)
        self.Qmax = -math.inf       # best value ever backed up through here
        self.value = None           # cached evaluation of this node's state
        self.terminal = terminal

    def mean(self) -> float:
        return self.W / self.N if self.N else 0.0


# --------------------------------------------------------------------------
# The engine.
# --------------------------------------------------------------------------
class MCTS:
    def __init__(self, game: SinglePlayerGame, c_puct: float = 1.4,
                 alpha: float = 0.5, fail_value: float = -1.0,
                 seed: int | None = None):
        self.game = game
        self.c_puct = c_puct
        self.alpha = alpha            # blend: alpha*max + (1-alpha)*mean
        self.fail_value = fail_value  # value assigned to failed transitions
        self.rng = random.Random(seed)
        # Incumbent: the best state ever *reliably* evaluated.
        self.best_value = -math.inf
        self.best_state = None
        self.evaluations = 0          # how many evaluate() calls were spent

    # ------------------------------------------------------------------ API
    def search(self, root_state, iterations: int) -> Node:
        root = Node(root_state, terminal=self.game.is_terminal(root_state))
        for _ in range(iterations):
            self._iterate(root)
        return root

    # ------------------------------------------------------------ internals
    def _iterate(self, root: Node):
        node, path = root, [root]

        # ---- 1. SELECTION: descend while fully expanded ----
        while True:
            if node.terminal:
                break
            if node.untried is None:                    # first visit here
                actions = self.game.legal_actions(node.state)
                if not actions:
                    node.terminal = True
                    break
                pri = self.game.priors(node.state, actions)
                # shuffle first so equal-prior actions are tie-broken
                # randomly, then stable-sort; pop() expands highest first
                self.rng.shuffle(actions)
                node.untried = sorted(actions, key=lambda a: pri.get(a, 0.0))
                node.action_priors = pri
            if node.untried:
                break                                   # something to expand
            node = self._select_child(node)             # PUCT descent
            path.append(node)

        # ---- 2. EXPANSION + 3. EVALUATION ----
        if not node.terminal and node.untried:
            action = node.untried.pop()
            child_state = self.game.transition(node.state, action)
            prior = node.action_priors.get(action, 0.0)
            if child_state is None:                     # failed transition
                child = Node(None, node, action, prior, terminal=True)
                child.value = self.fail_value
            else:
                child = Node(child_state, node, action, prior,
                             terminal=self.game.is_terminal(child_state))
                child.value = self._evaluate(child)
            node.children[action] = child
            path.append(child)
            value = child.value
        else:
            # Selection bottomed out at a terminal / exhausted node.
            if node.value is None:
                node.value = (self._evaluate(node) if node.state is not None
                              else self.fail_value)
            value = node.value

        # ---- 4. BACKUP (no sign flip: single player) ----
        for n in path:
            n.N += 1
            n.W += value
            if value > n.Qmax:
                n.Qmax = value

    def _evaluate(self, node: Node) -> float:
        v = self.game.evaluate(node.state)
        self.evaluations += 1
        # Update the incumbent only when v is trustworthy for THIS state:
        # always for ground-truth games, only at terminals for rollout games.
        if self.game.evaluation_is_ground_truth or node.terminal:
            if v > self.best_value:
                self.best_value, self.best_state = v, node.state
        return v

    def _select_child(self, node: Node) -> Node:
        sqrt_n = math.sqrt(node.N)
        # First-play urgency: an unvisited child inherits the parent's blend
        # so the prior term (not a fake Q of 0) decides among unvisited kids.
        parent_q = self._blend(node)
        best, best_score = None, -math.inf
        for child in node.children.values():
            q = self._blend(child) if child.N > 0 else parent_q
            u = self.c_puct * child.prior * sqrt_n / (1 + child.N)
            score = q + u
            if score > best_score:
                best, best_score = child, score
        return best

    def _blend(self, n: Node) -> float:
        if n.N == 0:
            return 0.0
        mx = n.Qmax if n.Qmax > -math.inf else n.mean()
        return self.alpha * mx + (1 - self.alpha) * n.mean()


# --------------------------------------------------------------------------
# Small reporting helper for demos.
# --------------------------------------------------------------------------
def describe_root(root: Node, top: int = 6) -> str:
    """Human-readable summary of the root's children, best-first."""
    rows = sorted(root.children.values(), key=lambda c: -c.N)[:top]
    lines = [f"root: N={root.N}"]
    for c in rows:
        lines.append(f"  {str(c.action):<28} N={c.N:<6} "
                     f"mean={c.mean():+.3f}  max={c.Qmax:+.3f}  "
                     f"prior={c.prior:.2f}")
    return "\n".join(lines)
