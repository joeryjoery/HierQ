"""
This file implements the canonical Hierarchical Q-learning algorithm by (Levy et al., 2019) along with
a relative action/ sub-goal space encoding, optionality for training goal-conditioned top-level policies,
and the optionality to opt between a shortest-path or binary goal-dependent reward function.

Note that the discount parameter should be in (0, 1) for the binary reward function and (0, 1] for the shortest path.

"""
from __future__ import annotations
import functools
import typing
import sys
from abc import ABC

import numpy as np
import gym
import tqdm

from ..interface import Agent
from .HierQ import TabularHierarchicalAgent

from .utils import CriticTable, GoalTransition, HierarchicalTrace

from mazelab_experimenter.utils import rand_argmax, get_pos
from mazelab_experimenter.utils import ravel_moore_index, ravel_neumann_index, unravel_moore_index, \
    unravel_neumann_index, manhattan_distance, chebyshev_distance, neumann_neighborhood_size, moore_neighborhood_size


class TabularHierarchicalAgentV3(TabularHierarchicalAgent, ABC):
    _ILLEGAL: int = -1
    _GOAL_PADDING: int = 1

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], discount: float = 0.95, alpha: float = 0.5,
                 beta: float = 0.5, epsilon: float = 0.1, alpha_decay: float = 0.0, beta_decay: float = 0.0,
                 greedy_options: bool = False, greedy_training: bool = False, sarsa: bool = False,
                 hindsight_goals: bool = True, legal_states: np.ndarray = None, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels, **kwargs)

        assert type(horizons) == int or (type(horizons) == list and len(horizons) == n_levels), \
            f"Incorrect specification of horizon lengths, " \
            f"this should either be a fixed integer or a list in ascending hierarchy order."

        # Learning parameters
        self.alpha_base = self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.beta_base = self.beta = beta
        self.beta_decay = beta_decay

        self.discount = discount
        self.sarsa = sarsa

        # Only atomic exploration by default for a given epsilon.
        self.epsilon = np.asarray([epsilon] + [0.] * (n_levels - 1), dtype=np.float32) \
            if type(epsilon) == float else np.asarray(epsilon, dtype=np.float32)
        # TODO: Infer n_levels based on a maximum horizon variable. Atomic horizons --> eg 3 9 27 81 MAX --> 4 levels
        self.horizons = np.full(n_levels, horizons, dtype=np.int32) \
            if type(horizons) == int else np.asarray(horizons, dtype=np.int32)
        self.atomic_horizons = [int(np.prod(self.horizons[:i])) for i in range(0, self.n_levels + 1)]

        # Agent parameterization configuration.
        self.hindsight_goals = hindsight_goals
        self.greedy_options = greedy_options
        self.greedy_training = greedy_training

        # Number of training episodes done.
        self.episodes = 0

        # Initialize the agent's state space, goal space and action space.
        self.S, self.S_legal = np.arange(np.prod(observation_shape)), legal_states
        self.A_flat = np.arange(self.n_actions)
        self.A_hierarchical = [
            self.create_lattice_neighborhoods(self.S, k) for k in self.atomic_horizons
        ]

        # Initialize Hierarchical Q-table as the Source map along with a goal-conditioned flat Q-table
        self.source = CriticTable(0, (len(self.S), len(self.S)), goal_conditioned=False)
        self.flat = CriticTable(0, (len(self.S), len(self.S), self.n_actions), goal_conditioned=True)

        self.source.reset()
        self.flat.reset()

    def create_lattice_neighborhoods(self, nodes: np.ndarray, k: int, norm: int) -> typing.List:
        distance = manhattan_distance if self.motion == Agent._NEUMANN_MOTION else chebyshev_distance

        # Sweep nodes O(|S|^2) to find all k-hop neighbors for each node on a lattice.
        states = list()
        for node in nodes:
            for neighbor in nodes:
                if node != neighbor:
                    if distance(*np.unravel_index([node, neighbor], shape=self.observation_shape)) < k:
                        states.append(neighbor)
        return states

    def _get_index(self, coord: typing.Tuple, dims: typing.Optional[typing.Tuple[int, int]] = None) -> int:
        return np.ravel_multi_index(coord, dims=self.observation_shape if dims is None else dims)

    def reset(self, full_reset: bool = False) -> None:
        self.clear_hierarchy(self.n_levels - 1 - int(not full_reset))
        self.alpha = self.alpha_base
        self.beta = self.beta_base
        self.episodes = 0
        self.source.reset()
        self.flat.reset()

    def terminate_option(self, level: int, state: int, goal: int, value: float) -> bool:
        if goal is None:
            return True

        if not self.greedy_options:
            return False

        # Extract pseudo Q(s, a) values based on the SR-values towards 'goal'.
        qs = self.source.table[self.A_hierarchical[level - 1][state], goal]
        return value < qs.max()

    def sample(self, state: np.ndarray, behaviour_policy: bool = True) -> int:
        """Sample an **Environment action** (not a goal) from the Agent. """
        assert self._level_goals[-1] is not None, "No environment goal is specified."

        s = self._get_index(get_pos(state))  # Ravelled state coordinate

        # Check if a goal has been achieved an horizon exceeded, and reset accordingly except the top-level.
        achieved = np.flatnonzero(s == self._level_goals)
        exceeded = np.flatnonzero(self._goal_steps[:-1] >= self.horizons[:-1])
        terminate = np.flatnonzero([self.terminate_option(i, s, self._level_goals[i], self._option_value[i])
                                    for i in range(1, len(self._level_goals))])

        if len(achieved) or len(exceeded) or len(terminate):
            self.clear_hierarchy(np.max(achieved.tolist() + exceeded.tolist() + terminate.tolist()))

        # Sample a new action and new goals by iterating and sampling actions from the top to the bottom policy.
        for lvl in reversed(range(self.n_levels)):
            if lvl == 0 or self._level_goals[lvl - 1] is None:
                # Get the (behaviour) policy action according to the current level.
                a, v = self.get_level_action(s=s, g=self._level_goals[lvl], level=lvl, explore=behaviour_policy)

                # Update internal agent state.
                self._goal_steps[lvl] += 1
                self._option_value[lvl] = v

                if lvl > 0:  # Set sampled action as goal for the 1-step lower level policy
                    self.set_goal(goal=a, level=lvl - 1)
                else:  # Return atomic action.
                    return a


class HierQV3(TabularHierarchicalAgentV3):
    _TEST_EPSILON: float = 0.05

    def policy_hook(function: typing.Callable) -> typing.Callable:
        """ Defines a decorator that enables the agent to follow a fixed action-trace on an atomic level. """
        @functools.wraps(function)
        def _policy_hook(self, level: int, *args, **kwargs) -> typing.Tuple[int, float]:
            if self._path and (not level):
                return self._path.pop(), None  # Return atomic action currently on top of the stack.
            return function(self, level=level, *args, **kwargs)
        return _policy_hook

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], discount: float = 0.95, decay: float = 0.9,
                 lr: float = 0.5, epsilon: float = 0.1, lr_decay: float = 0.0, greedy_options: bool = False,
                 relative_actions: bool = False, relative_goals: bool = False, universal_top: bool = False,
                 shortest_path_rewards: bool = False, sarsa: bool = False, stationary_filtering: bool = True,
                 hindsight_goals: bool = True, legal_states: np.ndarray = None, **kwargs) -> None:
        super().__init__(
            observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels, horizons=horizons,
            discount=discount, lr=lr, epsilon=epsilon, lr_decay=lr_decay, greedy_options=greedy_options,
            relative_actions=relative_actions, relative_goals=relative_goals, universal_top=universal_top,
            hindsight_goals=hindsight_goals, stationary_filtering=stationary_filtering,
            shortest_path_rewards=shortest_path_rewards, sarsa=sarsa, legal_states=legal_states, **kwargs
        )
        # Environment path to take overriding the agent policy (useful for debugging or visualization)
        self._path: typing.List[int] = None

        self.decay = decay

        self.strace = np.zeros(len(self.S), dtype=np.float32)
        self.qtrace = np.zeros((len(self.S), len(self.S), self.n_actions), dtype=np.float32)

        # Data structures for keeping track of an environment trace along with a deque that retains previous
        # states within the hierarchical policies' action horizon. Last atomic_horizon value should be large.
        self.trace = HierarchicalTrace(num_levels=self.n_levels, horizons=self.atomic_horizons)
        self.trace.reset()

    def _fix_policy(self, path: typing.List[int]) -> None:
        """ Fix the policy of the agent to strictly follow a given path *once*.
        Path should be ordered chronologically which is reversed in the form of an action-stack.
        """
        self._path = path[::-1].copy()

    @policy_hook
    def get_level_action(self, s: int, g: int, level: int, explore: bool = True) -> typing.Tuple[int, float]:
        """Sample a **Hierarchy action** (not an Environment action) from the Agent. """
        # Helper variables
        greedy = int((not level) or (not explore) or (np.random.rand() > self.epsilon[level]))

        if level == 0:  # Goal Conditioned Atomic policy --> for goal 'g' follow the optimal SR
            a = rand_argmax(self.flat.table[s, g, self.A_flat] * greedy)
            return a, self.flat.table[s, g, a]

        neighborhood = self.A_hierarchical[level - 1][s]  # Get all states within a k-hop node vicinity (on a lattice)
        if g in neighborhood:  # Directly move towards level goal if it is in reach.
            return g, 1.0

        # Multilevel Hierarchy derived from the Source map/ SR Matrix --> yields pseudo Q(s, a) values.
        qs = self.source.table[neighborhood, g]

        a = rand_argmax(qs * greedy)
        return a, self.source.table[a, g]

    def update_tables(self, trace: typing.Sequence[GoalTransition], **kwargs) -> None:
        """Update Q-Tables with given transition """
        s_next = trace[-1].next_state
        s, a, g = trace[-1].state, trace[-1].action, s_next

        # Apply sampled Bellman operator over all goals
        r = (self.S == g)
        gamma = self.discount * r
        pq = np.amax(self.flat.table[s_next, self.S], axis=-1)

        # delta ~ T*Q - Q
        delta = (r + gamma * pq) - self.flat.table[s, self.S, a]

        # Update eligibilities
        pis = (self.flat.table[s_next, self.S].max(axis=1) == self.flat.table[s_next, self.S, a])
        self.qtrace = (pis * gamma * self.decay) * self.qtrace
        self.qtrace[s, self.S, a] = 1.0

        # Watkin's Q(lambda) update for all goals simultaneously
        self.flat.table[s, self.S, a] += self.alpha * self.qtrace * delta

        # Update Source map using backward memory vector s.t., column i of S yields S_(:,i) = E[z]
        self.strace = self.strace * self.decay * self.discount
        self.strace[s] = 1

        self.source.table[self.S, s] += self.beta * (self.strace - self.source.table[self.S, s])

    def update(self, _env: gym.Env, level: int, state: int, goal_stack: typing.List[int],
               value_stack: typing.List[float]) -> typing.Tuple[int, bool]:
        """Train level function of Algorithm 2 HierQ by (Levy et al., 2019).
        """
        step, done = 0, False
        while (step < self.horizons[level]) and (state not in goal_stack) and (not done):
            # Sample an action at the current hierarchy level.
            a, v = self.get_level_action(s=state, g=goal_stack[-1], level=level)

            if level:  # Temporally extend action as a goal through recursion.
                s_next, done = self.update(_env=_env, level=level - 1, state=state, goal_stack=goal_stack + [a],
                                           value_stack=value_stack + [v])
            else:  # Atomic level: Perform a step in the actual environment, and perform critic-table updates.
                next_obs, r, done, meta = _env.step(a)
                s_next, terminal = self._get_index(meta['coord_next']), meta['goal_achieved']

                # Keep a memory of observed states to update Q-tables with hindsight.
                self.trace.add(GoalTransition(
                    state=state, goal=tuple(goal_stack), action=a, next_state=s_next, terminal=terminal, reward=r))

                # Update tables given the current trace of experience.
                self.update_tables(self.trace.raw)

            # Update state of control. Terminate level if new state is out of reach of current goal.
            state = s_next
            step += 1

        return state, done

    def update_training_variables(self) -> None:
        # Clear all trailing states in each policy's deque.
        self.trace.reset()

        if 0.0 < self.alpha_decay <= 1.0 and self.episodes:  # Decay according to lr_base * 1/(num_episodes ^ decay)
            self.alpha = self.alpha_base / (self.episodes ** self.alpha_decay)

        if 0.0 < self.beta_decay <= 1.0 and self.episodes:  # Decay according to lr_base * 1/(num_episodes ^ decay)
            self.beta = self.beta_base / (self.episodes ** self.beta_decay)

        # : add epsilon-decay?
        self.episodes += 1

    def train(self, _env: gym.Env, num_episodes: int, progress_bar: bool = False, **kwargs) -> None:
        """Train the HierQ agent through recursion with the `self.update` function.
        """
        iterator = (tqdm.trange(num_episodes, file=sys.stdout, desc="HierQ Training")
                    if progress_bar else range(num_episodes))

        for _ in iterator:
            # Reinitialize environment after each episode.
            state = self._get_index(get_pos(_env.reset()))

            # Set top hierarchy goal/ environment goal (only one goal-state is supported for now)
            goal_stack = [self._get_index(_env.unwrapped.maze.get_end_pos()[0])]

            # Reset memory of training episode.
            self.update_training_variables()

            done = False
            while (not done) and (state != goal_stack[0]):
                # Sample a goal as a temporally extended action and observe UMDP transition until env termination.
                state, done = self.update(
                    _env=_env, level=self.n_levels - 1, state=state, goal_stack=goal_stack, value_stack=[np.inf])

            # Cleanup environment variables
            _env.close()
