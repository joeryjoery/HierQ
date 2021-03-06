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


class TabularHierarchicalAgentV2(TabularHierarchicalAgent, ABC):
    _ILLEGAL: int = -1
    _GOAL_PADDING: int = 1

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], discount: float = 0.95,
                 lr: float = 0.5, epsilon: float = 0.1, lr_decay: float = 0.0, greedy_options: bool = False,
                 greedy_training: bool = False,
                 relative_actions: bool = False, relative_goals: bool = False, universal_top: bool = False,
                 shortest_path_rewards: bool = False, sarsa: bool = False, stationary_filtering: bool = True,
                 hindsight_goals: bool = True, legal_states: np.ndarray = None, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels, **kwargs)

        assert type(horizons) == int or (type(horizons) == list and len(horizons) == n_levels), \
            f"Incorrect specification of horizon lengths, " \
            f"this should either be a fixed integer or a list in ascending hierarchy order."

        # Learning parameters
        self.lr_base = self.lr = lr
        self.lr_decay = lr_decay
        self.discount = discount
        self.sarsa = sarsa
        # Only atomic exploration by default for a given epsilon.
        self.epsilon = np.asarray([epsilon] + [0.] * (n_levels - 1), dtype=np.float32) \
            if type(epsilon) == float else np.asarray(epsilon, dtype=np.float32)
        self.horizons = np.full(n_levels, horizons, dtype=np.int32) \
            if type(horizons) == int else np.asarray(horizons, dtype=np.int32)
        self.atomic_horizons = [int(np.prod(self.horizons[:i])) for i in range(0, self.n_levels + 1)]

        # Agent parameterization configuration.
        self.relative_actions = relative_actions
        self.relative_goals = relative_goals
        self.universal_top = universal_top
        self.hindsight_goals = hindsight_goals
        self.shortest_path_rewards = shortest_path_rewards
        self.stationary_filtering = stationary_filtering
        self.greedy_options = greedy_options
        self.greedy_training = greedy_training

        # Number of training episodes done.
        self.episodes = 0

        # Initialize the agent's state space, goal space and action space.
        self.S, self.S_legal = np.arange(np.prod(observation_shape)), legal_states
        self.G = [self.S] * n_levels
        self.A = [np.arange(self.n_actions)] + self.G[:-1]  # Defaults (A \equiv G \equiv S for pi_i, i > 0).

        # Cache coordinates of all states, goals, and actions for fast element-to-coordinate mapping.
        self.S_xy = np.asarray(np.unravel_index(self.S, observation_shape)).T
        self.G_xy = [self.S_xy] * n_levels
        self.A_xy = [None] + [self.S_xy] * (n_levels - 1)

        if self.relative_actions:  # Bound the action space for each policy level proportional to its horizon.
            self.A[1:], self.A_xy[1:] = self._to_neighborhood(radii=self.atomic_horizons[1:-1])

            # Yield a warning if the action-radii start to vastly exceed the environment dimensions.
            if np.sum(np.asarray(self.atomic_horizons[1:-1]) == max(self.observation_shape) // 2 - 1) > 1:
                print("Warning: multiple action spaces exceed the environment' size, perhaps use absolute goals?")

        if not self.universal_top:  # Do not condition top-level policy/ table on goals (more memory efficient).
            self.G[-1] = [0]

        if relative_goals or (not hindsight_goals):  # Bound the goal space for each policy level through masking
            self.G_radii = [((r + self._GOAL_PADDING) if hindsight_goals else 0) for r in self.atomic_horizons[1:-1]]
            _, goal_states = self._to_neighborhood(radii=self.G_radii, mask_center=False)

            self.goal_mask = list()
            for i in range(len(goal_states)):
                masks = list()
                for states in goal_states[i]:
                    mask = np.ones_like(self.G[i], dtype=np.bool)
                    pool = states[np.flatnonzero(states >= 0)]
                    mask[pool] = False

                    masks.append(mask)

                self.goal_mask.append(masks)

            self.goal_mask.append([np.zeros_like(self.G[-1])] * len(self.G[-1]))

        # Given all adjusted dimensions, initialize Q tables.
        self.critics = list()
        for i in range(self.n_levels):
            # Use pessimistic -Horizon_i initialization if using dense '-1' penalties, otherwise use 0s.
            init = -np.clip(self.atomic_horizons[i+1], 0, max(self.horizons)) * int(self.shortest_path_rewards)

            dims = (len(self.S), len(self.G[i]), len(self.A[i]))
            critic = CriticTable(init, dims, goal_conditioned=(i < self.n_levels - 1 or self.universal_top))
            critic.reset()

            self.critics.append(critic)

    def reward_func(self, r_mask: np.ndarray, bootstrap: np.ndarray) -> np.ndarray:
        if self.shortest_path_rewards:
            return (1 - r_mask) * (-1 + self.discount * bootstrap)
        else:
            return r_mask + (1 - r_mask) * self.discount * bootstrap

    def _to_neighborhood(self, radii: typing.List, mask_center: bool = True) -> typing.Tuple[typing.List, typing.List]:
        """ Correct the current absolute action space parameterization to a relative/ bounded action space. """
        relative_indices = [np.arange(
            neumann_neighborhood_size(r) if self.motion == Agent._NEUMANN_MOTION else moore_neighborhood_size(r)
        ) for r in radii]

        # Correct action-to-coordinate map to state space index for each level: f_i(A | S) -> S
        relative_coordinates = list()
        for i in range(1, self.n_levels):
            shifts = (   # First gather all relative coordinate displacements for each state.
                unravel_neumann_index(relative_indices[i - 1], radius=radii[i - 1], delta=True)
                if self.motion == Agent._NEUMANN_MOTION else
                unravel_moore_index(relative_indices[i - 1], radius=radii[i - 1], delta=True))

            neighborhood_states = list()
            for center in self.S_xy:
                coords = center + shifts  # All reachable coordinates from state 'center'

                # Mask out out of bound actions.
                mask = np.all((0, 0) <= coords, axis=-1) & np.all(coords < self.observation_shape, axis=-1)
                if mask_center:
                    mask[len(coords) // 2] = 0  # Do nothing action.

                states = TabularHierarchicalAgentV2._ILLEGAL * np.ones(len(coords))
                states[mask] = self._get_index(coords[mask].T)

                neighborhood_states.append(states.astype(np.int32))

            # Override current (absolute) mapping to their corrected displacements.
            relative_coordinates.append(neighborhood_states)

        return relative_indices, relative_coordinates

    @staticmethod
    def ravel_delta_indices(center_deltas: np.ndarray, r: int, motion: int) -> np.ndarray:
        if motion == Agent._MOORE_MOTION:
            return ravel_moore_index(center_deltas, radius=r, delta=True)
        else:
            return ravel_neumann_index(center_deltas, radius=r, delta=True)

    @staticmethod
    def inside_radius(a: np.ndarray, b: np.ndarray, r: int, motion: int) -> bool:
        """ Check whether the given arrays 'a' and 'b' are contained within the radius dependent on the motion. """
        return (manhattan_distance(a, b) if motion == Agent._NEUMANN_MOTION else chebyshev_distance(a, b)) < r

    def goal_reachable(self, level: int, state: int, goal: int) -> bool:
        if not self.relative_goals:
            return True
        return self.inside_radius(self.S_xy[state], self.S_xy[goal], r=self.atomic_horizons[level], motion=self.motion)

    def convert_action(self, level: int, reference: int, displaced: int, to_absolute: bool = False) -> int:
        if to_absolute:
            return self.A_xy[level][reference][displaced]
        else:
            return self.ravel_delta_indices((self.S_xy[displaced] - self.S_xy[reference])[None, :],
                                            r=self.atomic_horizons[level], motion=self.motion).item()

    def _get_index(self, coord: typing.Tuple, dims: typing.Optional[typing.Tuple[int, int]] = None) -> int:
        return np.ravel_multi_index(coord, dims=self.observation_shape if dims is None else dims)

    def reset(self, full_reset: bool = False) -> None:
        self.clear_hierarchy(self.n_levels - 1 - int(not full_reset))
        self.lr = self.lr_base
        self.episodes = 0
        for critic in self.critics:
            critic.reset()

    def terminate_option(self, level: int, state: int, goal: int, value: float) -> bool:
        if goal is None:
            return True

        if not self.greedy_options:
            return False

        goal = goal * int(self.critics[level].goal_conditioned)
        return value < self.critics[level].table[state, goal].max()

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


class HierQV2(TabularHierarchicalAgentV2):
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
                 horizons: typing.Union[typing.List[int]], discount: float = 0.95,
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
        gc = g * int(self.critics[level].goal_conditioned)  # TODO: g to relative index

        if level == 0:  # Epsilon greedy
            eps = self.epsilon[0] if explore else self._TEST_EPSILON
            if eps > np.random.rand():
                a = np.random.randint(self.n_actions)
            else:
                optimal = np.flatnonzero(self.critics[level].table[s, gc] == self.critics[level].table[s, gc].max())
                a = np.random.choice(optimal)

            return a, self.critics[level].table[s, gc, a]

        greedy = int((not explore) or (np.random.rand() > self.epsilon[level]))

        pref = g if level else None          # Goal-selection bias for shared max Q-values.
        if level and self.relative_actions:  # Action Preference: Absolute-to-Neighborhood for policy Action-Space
            pref = None
            if self.inside_radius(self.S_xy[s], self.S_xy[g], r=self.atomic_horizons[level], motion=self.motion):
                pref = self.ravel_delta_indices(center_deltas=np.diff([self.S_xy[s], self.S_xy[g]]).T,
                                                r=self.atomic_horizons[level], motion=self.motion).item()

        mask = np.ones_like(self.critics[level].table[s, gc], dtype=bool)
        if level:  # Prune/ correct action space by masking out-of-bound or illegal subgoals.
            if self.relative_actions:
                mask = self.A_xy[level][s] != HierQV2._ILLEGAL
            else:
                mask[s] = 0  # Do nothing is illegal.
            if self.S_legal is not None:
                pass  # TODO Illegal move masking.

        # Sample an action (with preference for the end-goal if goal-conditioned policy).
        action = rand_argmax(self.critics[level].table[s, gc] * greedy, preference=pref, mask=mask)
        value = self.critics[level].table[s, gc, action]

        if level and self.relative_actions:  # Neighborhood-to-Absolute for sampled action.
            action = self.convert_action(level, reference=s, displaced=action, to_absolute=True)

        return action, value

    def update_tables(self) -> None:
        # Always update atomic policy under the raw trace, update higher level policies only on actual transitions.
        self.update_table(level=0, horizon=1, trace=self.trace.raw)

        # Higher levels are either always updated or updates on actual environment transitions.
        raw = self.trace.raw[-1].degenerate
        if (not raw) or (not self.stationary_filtering):
            trace = self.trace.raw if raw else self.trace.transitions

            # Update Q tables at each level using built trace.
            for i in range(1, self.n_levels):
                self.update_table(level=i, horizon=self.trace.window(i, raw=raw), trace=trace)

    def update_table(self, level: int, horizon: int, trace: typing.Sequence[GoalTransition], **kwargs) -> None:
        """Update Q-Tables with given transition """
        s_next, end_pos = trace[-1].next_state, trace[-1].goal[0]
        for h in reversed(range(horizon)):
            s = trace[-(h + 1)].state
            a = s_next if level else trace[-1].action

            if level and self.relative_actions:  # Convert goal-action 'a' from Absolute-to-Neighborhood.
                a = self.convert_action(level, reference=s, displaced=s_next)

            # Hindsight action transition mask.
            if self.critics[level].goal_conditioned and (not self.hindsight_goals):
                # Single-Update: Only update goal-table for the current state-goal.
                goals, mask = (s_next, 1)
            else:
                # Multi-Update: Update all possible goals simultaneously for goal-conditioned tables.
                goals, mask = (self.G[level], self.G[level] == s_next) \
                    if self.critics[level].goal_conditioned else (self.G[level], end_pos == s_next)

            # Q-Learning update for each goal-state.
            bootstrap = np.amax(self.critics[level].table[s_next, goals], axis=-1)
            if self.sarsa:
                if (self.epsilon[level] * int(level == 0)) > np.random.rand():
                    Q_t = self.critics[level].table[s, goals, np.random.randint(len(self.A[level]))]
                # Expected SARSA target.
                # p = self.epsilon[level]
                # bootstrap = p * self.critics[level].table[s_next, goals].mean(axis=-1) + (1 - p) * bootstrap

            ys = self.reward_func(mask, bootstrap)
            delta = ys - self.critics[level].table[s, goals, a]
            if self.relative_goals or (not self.hindsight_goals):
                if self.critics[level].goal_conditioned:
                    # Bounded goals: Only update goal-table for the in-range goals
                    delta[self.goal_mask[level][s]] = 0

            self.critics[level].table[s, goals, a] += self.lr * delta

    def update(self, _env: gym.Env, level: int, state: int, goal_stack: typing.List[int],
               value_stack: typing.List[float]) -> typing.Tuple[int, bool]:
        """Train level function of Algorithm 2 HierQ by (Levy et al., 2019).
        """
        step, done = 0, False
        while (step < self.horizons[level]) and (state not in goal_stack) and (not done):
            if self.greedy_training:
                term = [self.terminate_option(i, state, goal_stack[self.n_levels - i - 1], value_stack[self.n_levels - i])
                        for i in range(level + 1, self.n_levels)]
                if any(term):
                    break

            # Sample an action at the current hierarchy level.
            a, v = self.get_level_action(s=state, g=goal_stack[-1], level=level)

            if level:  # Temporally extend action as a goal through recursion.
                s_next, done = self.update(_env=_env, level=level - 1, state=state, goal_stack=goal_stack + [a], value_stack=value_stack + [v])
            else:  # Atomic level: Perform a step in the actual environment, and perform critic-table updates.
                next_obs, r, done, meta = _env.step(a)
                s_next, terminal = self._get_index(meta['coord_next']), meta['goal_achieved']

                # Keep a memory of observed states to update Q-tables with hindsight.
                self.trace.add(GoalTransition(
                    state=state, goal=tuple(goal_stack), action=a, next_state=s_next, terminal=terminal, reward=r))

                # Update Q-tables given the current (global) trace of experience.
                # Always update the i = 0 table and either always update i > 0 tables or only on state-transitions.
                raw = self.trace.raw[-1].degenerate  # : s == s_next
                for i in range((self.n_levels if (not raw or (not self.stationary_filtering)) else 1)):
                    if (i == 0) or (not self.stationary_filtering):
                        self.update_table(level=i, horizon=self.trace.window(i, raw=True), trace=self.trace.raw)
                    else:
                        self.update_table(level=i, horizon=self.trace.window(i), trace=self.trace.transitions)

            # Update state of control. Terminate level if new state is out of reach of current goal.
            state = s_next
            step += 1

        return state, done

    def update_training_variables(self) -> None:
        # Clear all trailing states in each policy's deque.
        self.trace.reset()
        if 0.0 < self.lr_decay <= 1.0 and self.episodes:  # Decay according to lr_base * 1/(num_episodes ^ decay)
            self.lr = self.lr_base / (self.episodes ** self.lr_decay)
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
