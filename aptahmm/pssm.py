"""Position-specific scoring matrix over fixed-length windows of a seed set.

The seed sequences used to train the generator differ in length and are not
mutually alignable in any biologically meaningful way, so a conventional
profile PSSM over alignment columns is not defined for them. Instead the
PSSM here is built from every sliding window of the target length taken
across the (trimmed) seed set. It therefore encodes the positional base
composition of an L-nucleotide window of aptamer sequence space, and is used
to rank and filter generated candidates rather than to assert homology.
"""

from __future__ import annotations

import numpy as np

from .hmm import ALPHABET, N_SYMBOLS, SYMBOL_INDEX


class WindowPSSM:
    """Log-odds scoring matrix of shape (length, 4)."""

    def __init__(self, length: int, pseudocount: float = 1.0):
        if length < 1:
            raise ValueError("length must be >= 1")
        self.length = int(length)
        self.pseudocount = float(pseudocount)
        self.counts = np.zeros((self.length, N_SYMBOLS))
        self.background = np.full(N_SYMBOLS, 0.25)
        self.n_windows = 0
        self.matrix: np.ndarray | None = None

    def fit(self, sequences) -> "WindowPSSM":
        counts = np.full((self.length, N_SYMBOLS), self.pseudocount)
        bg = np.full(N_SYMBOLS, self.pseudocount)
        n_windows = 0
        for seq in sequences:
            s = seq.upper()
            for base in s:
                if base in SYMBOL_INDEX:
                    bg[SYMBOL_INDEX[base]] += 1
            for start in range(0, len(s) - self.length + 1):
                window = s[start : start + self.length]
                for pos, base in enumerate(window):
                    if base in SYMBOL_INDEX:
                        counts[pos, SYMBOL_INDEX[base]] += 1
                n_windows += 1
        if n_windows == 0:
            raise ValueError(
                "no seed sequence is long enough to yield a window of length "
                f"{self.length}"
            )
        self.counts = counts
        self.n_windows = n_windows
        self.background = bg / bg.sum()
        freqs = counts / counts.sum(axis=1, keepdims=True)
        self.matrix = np.log2(freqs / self.background[None, :])
        return self

    def score(self, sequence: str) -> float:
        """Total log-odds score (bits) of a sequence of the model length."""
        if self.matrix is None:
            raise RuntimeError("PSSM is not fitted")
        s = sequence.upper()
        if len(s) != self.length:
            raise ValueError(
                f"sequence length {len(s)} does not match PSSM length {self.length}"
            )
        return float(
            sum(self.matrix[pos, SYMBOL_INDEX[base]] for pos, base in enumerate(s))
        )

    def max_score(self) -> float:
        return float(self.matrix.max(axis=1).sum())

    def min_score(self) -> float:
        return float(self.matrix.min(axis=1).sum())

    def information_content(self) -> np.ndarray:
        """Per-position information content in bits (max 2 for a 4-letter alphabet)."""
        freqs = self.counts / self.counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.nansum(freqs * np.log2(freqs), axis=1)
        return 2.0 - entropy

    def to_dict(self) -> dict:
        return {
            "length": self.length,
            "pseudocount": self.pseudocount,
            "n_training_windows": self.n_windows,
            "alphabet": ALPHABET,
            "background_frequencies": self.background.tolist(),
            "log_odds_matrix_bits": self.matrix.tolist(),
            "per_position_information_bits": self.information_content().tolist(),
            "score_range_bits": [self.min_score(), self.max_score()],
        }
