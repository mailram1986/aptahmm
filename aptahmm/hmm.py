"""Discrete first-order hidden Markov model over the DNA alphabet.

Parameters are estimated by Baum-Welch (expectation-maximisation) in log
space, which is numerically stable for the short sequences and small
training sets typical of curated aptamer collections. Dirichlet
(Laplace-style) pseudocounts are added at every M-step so that no
transition or emission probability is driven to exactly zero by a sparse
training set.
"""

from __future__ import annotations

import numpy as np

ALPHABET = "ACGT"
SYMBOL_INDEX = {c: i for i, c in enumerate(ALPHABET)}
N_SYMBOLS = len(ALPHABET)

_NEG_INF = -np.inf


def encode(sequence: str) -> np.ndarray:
    """Map a DNA string to integer symbol indices."""
    seq = sequence.strip().upper().replace("U", "T")
    try:
        return np.array([SYMBOL_INDEX[c] for c in seq], dtype=np.int64)
    except KeyError as exc:  # pragma: no cover - guarded by validate_sequences
        raise ValueError(f"non-ACGT character in sequence: {exc.args[0]!r}") from exc


def decode(indices: np.ndarray) -> str:
    """Map integer symbol indices back to a DNA string."""
    return "".join(ALPHABET[int(i)] for i in indices)


def validate_sequences(sequences):
    """Return uppercase ACGT sequences, raising on any other character."""
    cleaned = []
    for i, seq in enumerate(sequences):
        s = seq.strip().upper().replace("U", "T")
        bad = set(s) - set(ALPHABET)
        if bad:
            raise ValueError(
                f"sequence {i + 1} contains non-ACGT characters: {sorted(bad)}"
            )
        if not s:
            raise ValueError(f"sequence {i + 1} is empty")
        cleaned.append(s)
    return cleaned


def _logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    """logsumexp without a SciPy dependency; -inf-safe."""
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


def _log_normalise(log_p: np.ndarray, axis=-1) -> np.ndarray:
    return log_p - np.expand_dims(_logsumexp(log_p, axis=axis), axis=axis)


class DiscreteHMM:
    """First-order HMM with ``n_states`` hidden states and 4 emission symbols.

    Estimation note
    ---------------
    With ``pseudocount > 0`` the M-step returns a maximum a posteriori rather
    than a maximum likelihood estimate (a symmetric Dirichlet prior on every
    row of the initial, transition and emission distributions). The classical
    guarantee that EM never decreases the data log-likelihood applies to the
    unpenalised estimator, so with pseudocounts the reported log-likelihood
    may decrease slightly in the final iterations as the estimate is pulled
    toward the prior. This is expected; ``history_`` records the trajectory so
    it can be inspected. Set a negligible pseudocount (e.g. 1e-9) to recover
    monotone behaviour, at the cost of zero-probability parameters on a sparse
    training set.

    Attributes
    ----------
    log_pi : (n_states,) log initial-state distribution
    log_A  : (n_states, n_states) log transition matrix, rows sum to 1
    log_B  : (n_states, 4) log emission matrix, rows sum to 1
    """

    def __init__(self, n_states: int, random_state: int | None = None):
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        self.n_states = int(n_states)
        self.random_state = random_state
        self.log_pi: np.ndarray | None = None
        self.log_A: np.ndarray | None = None
        self.log_B: np.ndarray | None = None
        self.n_iter_run_: int = 0
        self.log_likelihood_: float = float("nan")
        self.objective_: float = float("nan")
        self.history_: list[float] = []
        self.objective_history_: list[float] = []
        self.converged_: bool = False

    # ---------------------------------------------------------------- setup
    def _initialise(self, rng: np.random.Generator) -> None:
        k = self.n_states
        # Dirichlet draws keep the starting point asymmetric so that EM does
        # not sit at a symmetric saddle point.
        pi = rng.dirichlet(np.ones(k) * 5.0)
        A = rng.dirichlet(np.ones(k) * 5.0, size=k)
        B = rng.dirichlet(np.ones(N_SYMBOLS) * 2.0, size=k)
        self.log_pi = np.log(pi)
        self.log_A = np.log(A)
        self.log_B = np.log(B)

    def _log_prior(self, pseudocount: float) -> float:
        """Unnormalised log density of the symmetric Dirichlet prior.

        A pseudocount of ``c`` added in the M-step corresponds to a symmetric
        Dirichlet(c + 1) prior on each row, whose log density is ``c * sum(log
        theta)`` up to a constant that does not depend on the parameters.
        """
        if pseudocount <= 0.0:
            return 0.0
        return float(
            pseudocount
            * (self.log_pi.sum() + self.log_A.sum() + self.log_B.sum())
        )

    def objective(self, sequences, pseudocount: float) -> float:
        """Penalised log-likelihood (log posterior up to a constant).

        This is the quantity the M-step actually maximises, and the one whose
        convergence should be monitored when ``pseudocount > 0``.
        """
        return self.log_likelihood(sequences) + self._log_prior(pseudocount)

    @property
    def n_parameters(self) -> int:
        """Free parameters: initial dist + transitions + emissions."""
        k = self.n_states
        return (k - 1) + k * (k - 1) + k * (N_SYMBOLS - 1)

    # ------------------------------------------------------- forward/backward
    def _log_forward(self, obs: np.ndarray):
        t_len = obs.shape[0]
        la = np.empty((t_len, self.n_states))
        la[0] = self.log_pi + self.log_B[:, obs[0]]
        for t in range(1, t_len):
            la[t] = (
                _logsumexp(la[t - 1][:, None] + self.log_A, axis=0)
                + self.log_B[:, obs[t]]
            )
        return la, float(_logsumexp(la[-1]))

    def _log_backward(self, obs: np.ndarray) -> np.ndarray:
        t_len = obs.shape[0]
        lb = np.zeros((t_len, self.n_states))
        for t in range(t_len - 2, -1, -1):
            lb[t] = _logsumexp(
                self.log_A + (self.log_B[:, obs[t + 1]] + lb[t + 1])[None, :], axis=1
            )
        return lb

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        sequences,
        n_iter: int = 300,
        tol: float = 1e-5,
        pseudocount: float = 1.0,
        n_restarts: int = 5,
        verbose: bool = False,
    ) -> "DiscreteHMM":
        """Estimate parameters by Baum-Welch with multiple random restarts.

        The restart with the highest final log-likelihood is retained, which
        guards against EM converging to a poor local optimum - a real risk
        with a training set of only a few sequences.
        """
        sequences = validate_sequences(sequences)
        encoded = [encode(s) for s in sequences]
        rng = np.random.default_rng(self.random_state)

        best = None
        for restart in range(max(1, n_restarts)):
            self._initialise(rng)
            prev_objective = -np.inf
            history = []
            objective_history = []
            converged = False
            iters = 0
            for iteration in range(n_iter):
                iters = iteration + 1
                log_prior = self._log_prior(pseudocount)
                num_pi = np.full(self.n_states, pseudocount)
                num_A = np.full((self.n_states, self.n_states), pseudocount)
                num_B = np.full((self.n_states, N_SYMBOLS), pseudocount)
                total_ll = 0.0

                for obs in encoded:
                    la, ll = self._log_forward(obs)
                    lb = self._log_backward(obs)
                    total_ll += ll
                    log_gamma = la + lb - ll
                    gamma = np.exp(log_gamma)

                    num_pi += gamma[0]
                    for t in range(obs.shape[0] - 1):
                        log_xi = (
                            la[t][:, None]
                            + self.log_A
                            + (self.log_B[:, obs[t + 1]] + lb[t + 1])[None, :]
                            - ll
                        )
                        num_A += np.exp(log_xi)
                    for symbol in range(N_SYMBOLS):
                        mask = obs == symbol
                        if mask.any():
                            num_B[:, symbol] += gamma[mask].sum(axis=0)

                self.log_pi = _log_normalise(np.log(num_pi), axis=-1)
                self.log_A = _log_normalise(np.log(num_A), axis=-1)
                self.log_B = _log_normalise(np.log(num_B), axis=-1)

                # Monotonicity is guaranteed for the penalised objective,
                # which is what the M-step maximises, so convergence is tested
                # on that quantity rather than on the bare log-likelihood
                # (see the estimation note in the class docstring).
                objective = total_ll + log_prior
                history.append(total_ll)
                objective_history.append(objective)
                if verbose:
                    print(
                        f"  restart {restart} iter {iters:3d}  "
                        f"ll={total_ll:.4f}  objective={objective:.4f}"
                    )
                if iteration > 0 and abs(objective - prev_objective) < tol:
                    converged = True
                    break
                prev_objective = objective

            final_objective = self.objective(sequences, pseudocount)
            if best is None or final_objective > best["objective"]:
                best = {
                    "objective": final_objective,
                    "ll": self.log_likelihood(sequences),
                    "log_pi": self.log_pi.copy(),
                    "log_A": self.log_A.copy(),
                    "log_B": self.log_B.copy(),
                    "history": history,
                    "objective_history": objective_history,
                    "iters": iters,
                    "converged": converged,
                }

        self.log_pi = best["log_pi"]
        self.log_A = best["log_A"]
        self.log_B = best["log_B"]
        self.log_likelihood_ = best["ll"]
        self.objective_ = best["objective"]
        self.history_ = best["history"]
        self.objective_history_ = best["objective_history"]
        self.n_iter_run_ = best["iters"]
        self.converged_ = best["converged"]
        if not self.converged_:
            import warnings

            warnings.warn(
                f"EM reached the iteration cap ({n_iter}) without meeting the "
                f"tolerance {tol:g} in the best of {n_restarts} restarts; "
                "raise n_iter or loosen tol before quoting these parameters",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    # -------------------------------------------------------------- scoring
    def log_likelihood(self, sequences) -> float:
        sequences = validate_sequences(sequences)
        return float(sum(self._log_forward(encode(s))[1] for s in sequences))

    def bic(self, sequences) -> float:
        """Bayesian information criterion; lower is better."""
        sequences = validate_sequences(sequences)
        n_obs = sum(len(s) for s in sequences)
        return -2.0 * self.log_likelihood(sequences) + self.n_parameters * np.log(n_obs)

    def aic(self, sequences) -> float:
        """Akaike information criterion; lower is better."""
        return -2.0 * self.log_likelihood(sequences) + 2.0 * self.n_parameters

    # ------------------------------------------------------------- sampling
    def sample(self, length: int, rng: np.random.Generator) -> str:
        """Draw one sequence of the requested length from the fitted model."""
        if self.log_pi is None:
            raise RuntimeError("model is not fitted")
        if length < 1:
            raise ValueError("length must be >= 1")
        pi = np.exp(self.log_pi)
        A = np.exp(self.log_A)
        B = np.exp(self.log_B)
        state = rng.choice(self.n_states, p=pi / pi.sum())
        out = []
        for _ in range(length):
            out.append(ALPHABET[rng.choice(N_SYMBOLS, p=B[state] / B[state].sum())])
            state = rng.choice(self.n_states, p=A[state] / A[state].sum())
        return "".join(out)

    def sample_many(self, n: int, length: int, rng: np.random.Generator) -> list[str]:
        return [self.sample(length, rng) for _ in range(n)]

    # --------------------------------------------------------------- export
    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "model_order": 1,
            "n_symbols": N_SYMBOLS,
            "alphabet": ALPHABET,
            "n_free_parameters": self.n_parameters,
            "log_likelihood": self.log_likelihood_,
            "penalised_objective": self.objective_,
            "em_iterations": self.n_iter_run_,
            "converged": self.converged_,
            "initial_distribution": np.exp(self.log_pi).tolist(),
            "transition_matrix": np.exp(self.log_A).tolist(),
            "emission_matrix": np.exp(self.log_B).tolist(),
        }
