"""
This file implements the canonical Hierarchical Q-learning algorithm by (Levy et al., 2019) along with
a relative action/ sub-goal space encoding, optionality for training goal-conditioned top-level policies,
and the optionality to opt between a shortest-path or binary goal-dependent reward function.

Note that the discount parameter should be in (0, 1) for the binary reward function and (0, 1] for the shortest path.

"""
from __future__ import annotations
import functools
import sys
import warnings
import typing
from typing import Tuple, Optional, List, Union
from abc import ABC

import numpy as np
import gym
import tqdm

from ..interface import Agent
from .HierQ import TabularHierarchicalAgent

from .utils import CriticTable, GoalTransition, HierarchicalTrace

from mazelab_experimenter.utils import rand_argmax, get_pos
from mazelab_experimenter.utils import unravel_moore_index, unravel_neumann_index, \
    neumann_neighborhood_size, moore_neighborhood_size


class TabularHierarchicalAgentV3(TabularHierarchicalAgent, ABC):
    _ILLEGAL: int = -1
    _GOAL_PADDING: int = 1

    def __init__(self, observation_shape: Tuple, n_actions: int, n_levels: int,
                 horizons: Union[List[int]], discount: float = 0.95,
                 lr: float = 0.5, epsilon: float = 0.1, lr_decay: float = 0.0, greedy_options: bool = False,
                 legal_states: np.ndarray = None, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels, **kwargs)

        assert type(horizons) == int or (type(horizons) == list and len(horizons) == n_levels), \
            f"Incorrect specification of horizon lengths, " \
            f"this should either be a fixed integer or a list in ascending hierarchy order."

        # Learning parameters
        self.lr_base = self.lr = lr
        self.lr_decay = lr_decay
        self.discount = discount

        # Only atomic exploration by default for a given epsilon unless explicitly provided per level.
        self.epsilon = np.asarray([epsilon] + [0.] * (n_levels - 1), dtype=np.float32) \
            if type(epsilon) == float else np.asarray(epsilon, dtype=np.float32)

        self.horizons = np.full(n_levels, horizons, dtype=np.int32) \
            if type(horizons) == int else np.asarray(horizons, dtype=np.int32)

        # Neighborhood sizes
        self.atomic_horizons = [int(np.prod(self.horizons[:i])) for i in range(0, self.n_levels + 1)]

        # Agent parameterization configuration.
        self.greedy_options = greedy_options

        # Number of training episodes done.
        self.episodes = 0

        # Initialize the agent's state space, goal space and action space.
        self.S, self.S_legal = np.arange(np.prod(observation_shape)), legal_states
        self.S_xy = np.asarray(np.unravel_index(self.S, observation_shape)).T
        self.G = self.S

        # Neighborhoods U are slices (lists of indices) of S.
        # self.U is a subset of self.neighbors based on observed states/ experience!
        # self.neighbors serves as an upper-bound/ check for monitoring self.U!
        self.neighbors = self._generate_neighborhoods(radii=self.atomic_horizons[1:-1])
        self.U = [[list() for _ in range(len(self.S))] for __ in range(self.n_levels - 1)]
        self.U_seen = [[set() for _ in range(len(self.S))] for __ in range(self.n_levels - 1)]

        self.A = [list(range(self.n_actions))] + self.U

        # Yield a warning if the action-radii start to vastly exceed the environment dimensions.
        if np.sum(np.asarray(self.atomic_horizons[1:-1]) == max(self.observation_shape) // 2 - 1) > 1:
            warnings.warn("Multiple neighborhoods equal the environment size. This wastes computation time.")

        # Initialize goal-conditioned Q-table for atomic level and pseudo Q-table for the hierarchical levels.
        self.critic_flat = CriticTable(0, (len(self.S), self.n_actions, len(self.G)), goal_conditioned=True)
        self.critic_flat.reset()  # Q(s, a, g)

        self.critics_hier = list()
        for _ in range(self.n_levels - 1):
            c = CriticTable(0, (len(self.S), len(self.G)), goal_conditioned=True)
            c.reset()  # Q_k(s, a, g) = S_pi_g[U_k(s, a), g]

            self.critics_hier.append(c)

    def _generate_neighborhoods(self, radii: List, mask_center: bool = True) -> List:
        """ Correct the current absolute action space parameterization to a relative/ bounded action space. """
        indices = [np.arange(
            neumann_neighborhood_size(r) if self.motion == Agent._NEUMANN_MOTION else moore_neighborhood_size(r)
        ) for r in radii]

        # Correct action-to-coordinate map to state space index for each level: f_i(A | S) -> S
        coordinate_map = list()
        for i in range(1, self.n_levels):
            shifts = (   # First gather all relative coordinate displacements for each state.
                unravel_neumann_index(indices[i - 1], radius=radii[i - 1], delta=True)
                if self.motion == Agent._NEUMANN_MOTION else
                unravel_moore_index(indices[i - 1], radius=radii[i - 1], delta=True)
            )

            neighborhood_xy = list()
            for center in self.S_xy:
                coords = center + shifts  # All reachable coordinates from state 'center'

                # Remove undefined states/ out out of bound actions.
                mask = np.all((0, 0) <= coords, axis=-1) & np.all(coords < self.observation_shape, axis=-1)
                if mask_center:
                    mask[len(coords) // 2] = 0  # Do nothing action.

                xy_s = self._get_index(coords[mask].T)
                neighborhood_xy.append(xy_s.astype(np.int32))

            coordinate_map.append(neighborhood_xy)

        return coordinate_map

    def _get_index(self, coord: Tuple, dims: Optional[Tuple[int, int]] = None) -> Union[int, np.ndarray]:
        """ Convert an (x, y) coordinate to a flattened index (or an array/ list of). """
        return np.ravel_multi_index(coord, dims=self.observation_shape if dims is None else dims)

    def update_neighborhood(self, level: int, state: int, neighbor_state: int, bidirectional: bool = False):
        if state == neighbor_state:
            return

        if neighbor_state not in self.U_seen[level][state]:
            self.U_seen[level][state].add(neighbor_state)
            self.U[level][state].append(neighbor_state)

        if bidirectional:
            self.update_neighborhood(level, state=neighbor_state, neighbor_state=state, bidirectional=False)

    def reset(self, full_reset: bool = False) -> None:
        self.clear_hierarchy(self.n_levels - 1 - int(not full_reset))
        self.lr = self.lr_base
        self.episodes = 0

        self.critic_flat.reset()
        for c in self.critics_hier:
            c.reset()

    def terminate_option(self, level: int, state: int, goal: int, value: float) -> bool:
        if goal is None or goal == -1:
            return True
        elif not self.greedy_options:
            return False

        n_k = self.U[level - 1][state]

        if not len(n_k):  # No other goals/ options available.
            return False

        return value < self.critics_hier[level - 1].table[n_k, goal].max()
            
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
            self.clear_hierarchy(np.max(list(achieved) + list(exceeded) + list(terminate)))

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
    _TEST_EPSILON: float = 0.01

    def policy_hook(function: typing.Callable) -> typing.Callable:
        """ Defines a decorator that enables the agent to follow a fixed action-trace on an atomic level. """
        @functools.wraps(function)
        def _policy_hook(self, level: int, *args, **kwargs) -> Tuple[Optional[int], None]:
            if self._path and (not level):
                return self._path.pop(), None  # Return atomic action currently on top of the stack.
            return function(self, level=level, *args, **kwargs)
        return _policy_hook

    def __init__(self, observation_shape: Tuple, n_actions: int, n_levels: int,
                 horizons: Union[List[int]], discount: float = 0.95,
                 lr: float = 0.5, epsilon: float = 0.1, lr_decay: float = 0.0, greedy_options: bool = False,
                 shortest_path_rewards: bool = False, legal_states: np.ndarray = None, **kwargs) -> None:
        super().__init__(
            observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels, horizons=horizons,
            discount=discount, lr=lr, epsilon=epsilon, lr_decay=lr_decay, greedy_options=greedy_options,
            shortest_path_rewards=shortest_path_rewards,
            legal_states=legal_states, **kwargs
        )
        # Environment path to take overriding the agent policy (useful for debugging or visualization)
        self._path: Optional[List[int]] = None

        self.trace = HierarchicalTrace(num_levels=self.n_levels, horizons=self.atomic_horizons)
        self.trace.reset()

    def _fix_policy(self, path: List[int]) -> None:
        """ Fix the policy of the agent to strictly follow a given path *once*.
        Path should be ordered chronologically which is reversed in the form of an action-stack.
        """
        self._path = path[::-1].copy()

    @policy_hook
    def get_level_action(self, s: int, g: int, level: int, explore: bool = True) -> Tuple[int, float]:
        """Sample a **Hierarchy action** (not an Environment action) from the Agent.

        Note: s and g are absolute indices (i.e., environment indices; not relative table indices).
        """
        if not level:  # Epsilon greedy
            if g == -1:
                return np.random.randint(self.n_actions), 0  # Random walk if no concrete goal is provided.

            eps = self.epsilon[0] if explore else self._TEST_EPSILON
            a = rand_argmax(self.critic_flat.table[s, :, g] * int(eps < np.random.rand()))

            return a, self.critic_flat.table[s, a, g]

        # Slice Q over the neighborhood of states conditioned on reaching goal g.
        n_k = self.U[level - 1][s]

        if not len(n_k) or g == -1:  # Random walk if no concrete goal is provided, or no goal-states available.
            return -1, 0

        qs = self.critics_hier[level - 1].table[n_k, g]

        # Sample a neighborhood-action epsilon-greedy and return its absolute (new) goal-state + value
        a = rand_argmax(qs * int((not explore) or (np.random.rand() > self.epsilon[level])))

        return n_k[a], qs[a]

    def update_flat(self, transition: GoalTransition):
        s, a, s_next = transition.state, transition.action, transition.next_state

        # Cast TD-error and Bellman update across all goals simultaneously.
        mask = self.G == s_next  # == R_t+1
        TQ = mask + (1 - mask) * self.discount * self.critic_flat.table[s_next].max(axis=0)

        # Cast TD-error over all goals
        self.critic_flat.table[s, a] += self.lr * (TQ - self.critic_flat.table[s, a])

    def update_hierarchy(self, transition: GoalTransition, trace: HierarchicalTrace):
        mask = self.G == transition.next_state  # == R_t+1

        for i in range(self.n_levels - 1):
            c = self.critics_hier[i]

            n_k = self.U[i][transition.next_state]

            bootstrap = c.table[n_k].max(axis=0) if len(n_k) else 0.0

            TQ = mask + (1 - mask) * self.discount * bootstrap

            c.table[transition.next_state] += self.lr * (TQ - c.table[transition.next_state])

    def update(self, _env: gym.Env, level: int, state: int, goal_stack: List[int],
               value_stack: List[float]) -> Tuple[int, bool]:
        """Modified train level function of Algorithm 2 HierQ by (Levy et al., 2019).
        """
        step, done = 0, False
        while (step < self.horizons[level]) and (state not in goal_stack) and (not done):
            # Sample an action at the current hierarchy level.
            a, v = self.get_level_action(s=state, g=goal_stack[-1], level=level)

            if level:  # Temporally extend action as a goal through recursion.
                s_next, done = self.update(_env=_env, level=level - 1, state=state,
                                           goal_stack=goal_stack + [a], value_stack=value_stack + [v])
            else:  # Atomic level: Perform a step in the actual environment, and perform critic-table updates.
                next_obs, r, done, meta = _env.step(a)
                s_next, terminal = self._get_index(meta['coord_next']), meta['goal_achieved']

                # Keep a memory of observed states to update Q-tables with hindsight.00
                transition = GoalTransition(
                    state=state, goal=tuple(goal_stack), action=a, next_state=s_next, terminal=terminal, reward=r)
                self.trace.add(transition)

                # Update Q-tables given the current (global) trace of experience.
                # Always update the i = 0 table and either always update i > 0 tables or only on state-transitions.
                self.update_flat(transition)
                if (self.n_levels > 1) and (not transition.degenerate):
                    self.update_hierarchy(transition, self.trace)

                    # Add all trailing states (within the hierarchy time-window) as possible goal-states/ actions.
                    for i in range(self.n_levels - 1):
                        for t in self.trace.transitions[-(self.trace.window(i + 1) + 1):]:
                            self.update_neighborhood(i, t.state, transition.next_state, bidirectional=True)

            # Update state of control. Terminate level if new state is out of reach of current goal.
            state = s_next
            step += 1

        return state, done

    def update_training_variables(self) -> None:
        self.trace.reset()
        # Clear all trailing states in each policy's deque.
        if 0.0 < self.lr_decay <= 1.0 and self.episodes:  # Decay according to lr_base * 1/(num_episodes ^ decay)
            self.lr = self.lr_base / (self.episodes ** self.lr_decay)
        # TODO: add epsilon-decay?
        self.episodes += 1

    def train(self, _env: gym.Env, num_episodes: int, progress_bar: bool = False, **kwargs) -> None:
        """Train the HierQS agent through recursion with the `self.update` function.
        """
        iterator = (tqdm.trange(num_episodes, file=sys.stdout, desc="HierQS Training")
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
                # Sample a goal as a temporally extended action and observe transition until env termination.
                state, done = self.update(
                    _env=_env, level=self.n_levels - 1, state=state, goal_stack=goal_stack, value_stack=[np.inf])

            # Cleanup environment variables
            _env.close()
