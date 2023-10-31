# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## MONTE CARLO CONTROL ALGORITHM
Step 1:
Initialize Q-values, state-value function, and the policy.

Step 2:
Interact with the environment to collect episodes using the current policy.

Step 3:
For each time step within episodes, calculate returns (cumulative rewards) and update Q-values.

Step 4:
Update the policy based on the improved Q-values.

Step 5:
Repeat steps 2-4 for a specified number of episodes or until convergence.

Step 6:
Return the optimal Q-values, state-value function, and policy.



## MONTE CARLO CONTROL FUNCTION
```
import numpy as np
from collections import defaultdict

def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):

    nS, nA = env.observation_space.n, env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    V = defaultdict(float)
    pi = defaultdict(lambda: np.random.choice(nA))  
    Q_track = []
    pi_track = []
    for episode in range(n_episodes):
        epsilon = max(init_epsilon * (epsilon_decay_ratio ** episode), min_epsilon)
        alpha = max(init_alpha * (alpha_decay_ratio ** episode), min_alpha)  
        trajectory = generate_trajectory(select_action, Q, epsilon, env, max_steps)
        n = len(trajectory)
        G = 0  
        for t in range(n - 1, -1, -1):
            state, action, reward, _, _ = trajectory[t]
            G = gamma * G + reward
            if first_visit and (state, action) not in [(s, a) for s, a, _, _, _ in trajectory[:t]]:
                Q[state][action] += alpha * (G - Q[state][action])
                V[state] = np.max(Q[state])
                pi[state] = np.argmax(Q[state])
        Q_track.append(Q.copy())
        pi_track.append(pi.copy)
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
![275240017-9a8197f2-c405-42bf-aa45-84e8082b35fc](https://github.com/Fawziya20/monte-carlo-control/assets/75235022/afe85bf2-c935-486b-87c3-0f2962c4b03e)


## RESULT:
Monte Carlo Control successfully learned an optimal policy for the specified environment.
