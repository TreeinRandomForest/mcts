"""
session1_bandits.py -- multi-armed bandits: exploration vs exploitation
with no tree.

K slot machines, each paying Gaussian rewards with an unknown mean. Pull
T times; regret = how much you lost vs always pulling the (unknown) best
arm. Three policies:

  * eps-greedy(0.1)  -- explore 10% of the time (forever) i.e. pick random arm 10% of the time
  * eps-greedy(0.01) -- explore 1% of the time
  * UCB1             -- pull argmax  Q(a) + sqrt(2 ln t / N(a)) - first term is exploitative, second term is explorative

UCB1: exploration is not fixed (like eps-greedy) but dynamic based on past
behavior and received rewards. MCTS does UCB1 at each node in the tree 
(see: http://ggp.stanford.edu/readings/uct.pdf)  

Kernel problem:
Choose one of the available optimization strategies for a given kernel.

"""

from __future__ import annotations

import math
import random


def eps_greedy(eps: float):
    def policy(_t, N, Q, rng):
        '''
        N = #actions/strategies
        Q[a] = average reward observed for action a
        t = total number of time-steps executed so far (unused var)
        '''

        if rng.random() < eps:
            return rng.randrange(len(N))
        m = max(Q)
        
        #don't use argmax (which breaks ties by picking first element). instead want to pick uniformly among ties
        return rng.choice([a for a in range(len(Q)) if Q[a] == m]) 
    return policy


def ucb1(t, N, Q, rng):
    '''
    N = #actions/strategies
    Q[a] = average reward observed for action a
    t = total number of time-steps executed so far (unused var)
    '''

    for a in range(len(N)):
        if N[a] == 0: #every arm should be played at least once
            return a
    
    #pick action with max UCB1 value across N actions
    return max(range(len(N)), key=lambda a: Q[a] + math.sqrt(2 * math.log(t) / N[a]))


def run(policy, K=10, T=3000, seed=0):
    rng = random.Random(seed)
    means = [rng.gauss(0, 1) for _ in range(K)] #randomly choose gaussian means - can use any other strategy to pick these
    best = max(means)
    
    
    N, Q = [0] * K, [0.0] * K #N=counts for each action, Q=avg reward for each action
    regret, curve = 0.0, [] #regret is a cumulative metric
    
    for t in range(1, T + 1): #T = total budget of trials. Total regret bounded by T*(best mean - worst mean)
        a = policy(t, N, Q, rng) #pick action according to epsilon greedy or ucb1

        r = rng.gauss(means[a], 1.0) #pull arm i.e. sample from gaussian arm. Std dev fixed to 1.0 for all arms

        N[a] += 1 #update count for this action
        Q[a] += (r - Q[a]) / N[a] #update average reward for this arm

        regret += best - means[a]
        if t % (T // 10) == 0: #print 10 messages and log regret growth
            curve.append(regret)

    return curve


if __name__ == "__main__":
    T, runs = 3000, 200 #200 runs i.e. repeat each experiment 200 times

    policies = [("eps-greedy 0.10", eps_greedy(0.10)),
                ("eps-greedy 0.01", eps_greedy(0.01)),
                ("UCB1           ", ucb1)]

    print(f"{runs} runs, {T} pulls, 10 arms. Cumulative regret (lower = better):\n")
    print("policy            " + "".join(f"t={T//10*(i+1):<7}" for i in range(0, 10, 3)) + "final")

    results = {}
    for name, pol in policies:
        curves = [run(pol, T=T, seed=s) for s in range(runs)]

        avg = [sum(c[i] for c in curves) / runs for i in range(10)] #across runs

        results[name] = avg

        print(f"{name}   " + "".join(f"{avg[i]:<9.0f}" for i in range(0, 10, 3)) + f"{avg[-1]:.0f}")

    #pretty ascii printing
    print("\nregret growth, first -> last checkpoint (ascii):")
    scale = max(v for a in results.values() for v in a)
    for name, avg in results.items():
        print(f"  {name}  " + " ".join("#" * max(1, round(12 * v / scale)) for v in avg[::3]))

    # ----------------------------------------------------------------------
    # Talking points:
    #  * eps-greedy never stops paying the exploration tax: linear regret.
    #    UCB1's tax shrinks as ln(t)/N -- logarithmic regret.
    #  * eps=0.01 sometimes beats UCB1 early and sometimes locks onto a
    #    wrong arm forever; rerun with different `runs` and discuss variance.
    #  * Preview: a game tree is a bandit at every node, where each arm's
    #    payout distribution is defined by the search happening below it.
    # ----------------------------------------------------------------------
