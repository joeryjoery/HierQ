from __future__ import annotations
import typing

import numpy as np
from .HierQV3 import HierQV3

from .utils import GoalTransition


class HierQS(HierQV3):

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], lr: float = 0.5, epsilon: float = 0.1,
                 discount: float = 0.95, decay: float = 0.9, greedy_options: bool = False,
                 legal_states: np.ndarray = None, **kwargs) -> None:
        super().__init__(
            observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels,
            horizons=horizons, lr=lr, epsilon=epsilon, discount=discount, greedy_options=greedy_options,
            legal_states=legal_states, **kwargs
        )
        # Lambda value for decaying compound returns inside the eligibility trace.
        self.decay = decay

        # Flat eligibility
        self.z = np.zeros((len(self.S), self.n_actions, len(self.G)), dtype=np.float32)
        # Hierarchical state-eligibility for each level (to correct decay of the trace).
        self.x = [np.zeros(len(self.S), dtype=np.float32) for i in range(self.n_levels - 1)]
        # Hierarchical state-correction for each level and for each maximum horizon.
        self.w = [[np.ones((len(self.S), len(self.G)), dtype=np.float32)
                   for _ in range(self.atomic_horizons[i + 1])]
                  for i in range(self.n_levels - 1)]

    def update_flat(self, transition: GoalTransition):
        s, a, s_next = transition.state, transition.action, transition.next_state

        # Cast TD-error and Bellman update across all goals simultaneously.
        mask = self.G == s_next  # == R_t+1
        TQ = mask + (1 - mask) * self.discount * self.critic_flat.table[s_next].max(axis=0)
        delta = TQ - self.critic_flat.table[s, a]

        # Policy correction: greedy --> Watkin's Q(lambda)
        pi = (self.critic_flat.table[s, a] == self.critic_flat.table[s].max(axis=0))

        self.z *= self.discount * self.decay * pi
        self.z[..., s] = 0.0  # Cut trace behind the previously achieved state-goal.
        self.z[s, a] = 1.0    # Replacing trace on the performed (s, a) pair.

        # Cast TD-error over all goals
        self.critic_flat.table += self.lr * delta * self.z

    def update_hierarchy(self, transition, trace):
        s_next = transition.next_state
        mask = self.G == s_next  # == R_t+1

        for i in range(self.n_levels - 1):
            # Helper variables
            c = self.critics_hier[i]
            h_atomic = self.trace.window(i + 1)
            h = (len(trace) - 1) % h_atomic
            s_h = trace[-h_atomic].state

            # Update state-recencies 'x'.
            self.x[i] *= self.decay * (self.discount ** (1.0 / self.atomic_horizons[i + 1]))
            self.x[i][s_next] = 1.0

            # Construct a per-goal mask for a == s_next --> greedy action state s_h.
            n_k_prev = self.U[i][s_h]
            q_greedy = c.table[n_k_prev].max(axis=0) if len(n_k_prev) else c.table[s_next]
            pi_h = (c.table[s_next] == q_greedy)

            # Update policy corrections 'w'.
            self.w[i][h][..., s_h] = 0.0   # Cut state-trace behind previously achieved state-goal.
            self.w[i][h][:, ~pi_h] = 0.0   # Policy correction: Cut goal-traces
            self.w[i][h][s_next, :] = 1.0  # Add state to all goal-traces.

            # Compute Update Target.
            n_k = self.U[i][s_next]
            bootstrap = c.table[n_k].max(axis=0) if len(n_k) else 0.0
            TQ = mask + (1 - mask) * self.discount * bootstrap
            delta = TQ - c.table[s_next]

            # Construct eligibilities 'z' by casting 'x' column-wise over 'w'.
            z = self.w[i][h] * self.x[i][:, None]

            # Update table by casting TD-errors (delta) row-wise over eligibility trace 'z'.
            c.table += self.lr * z * delta

            # Add 'next_state' as a valid action to all separate eligibility traces.
            for wj in self.w[i]:
                wj[s_next] = 1.0

    def update_training_variables(self) -> None:
        """ Additionally reset eligibility traces along with parent functionality. """
        super().update_training_variables()
        self.trace.reset()

        self.z[...] = 0.0
        for i in range(self.n_levels - 1):
            self.x[i][...] = 0.0
            for h_w in self.w[i]:
                h_w[...] = 1.0
