#!/usr/bin/env python3
"""Diagnostic: per-sequence posterior occupancy of each hidden state.

With a small seed set a hidden state can be driven almost entirely by a
single unusual sequence, which makes the fitted model unstable and its
interpretation misleading. This script reports, for each seed, the mean
posterior probability of occupying each state, so such cases are visible.

Usage:
    python scripts/state_occupancy.py data/seeds_ssdna_aptamers.fasta --states 2
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from aptahmm import DiscreteHMM  # noqa: E402
from aptahmm.hmm import encode  # noqa: E402
from aptahmm.io import read_sequences  # noqa: E402
from aptahmm.preprocess import trim_flanks  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seeds")
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--random-seed", type=int, default=20260101)
    ap.add_argument("--no-trim", dest="trim", action="store_false")
    ap.add_argument("--em-iterations", type=int, default=500)
    ap.add_argument("--em-restarts", type=int, default=5)
    ap.set_defaults(trim=True)
    args = ap.parse_args(argv)

    sequences, identifiers = read_sequences(args.seeds)
    if args.trim:
        sequences, _, _ = trim_flanks(sequences)

    model = DiscreteHMM(args.states, random_state=args.random_seed).fit(
        sequences, n_iter=args.em_iterations, n_restarts=args.em_restarts
    )

    print(f"fitted {args.states}-state model, logL={model.log_likelihood_:.3f}, "
          f"converged={model.converged_}\n")
    print("emission probabilities")
    for i, row in enumerate(np.exp(model.log_B)):
        print(f"  state {i + 1}: A={row[0]:.3f} C={row[1]:.3f} "
              f"G={row[2]:.3f} T={row[3]:.3f}")
    print("\ntransition probabilities")
    for i, row in enumerate(np.exp(model.log_A)):
        print(f"  from state {i + 1}: " + " ".join(f"{p:.3f}" for p in row))

    header = f"\n{'id':<20}{'len':>5}" + "".join(
        f"{'state' + str(i + 1) + '%':>10}" for i in range(args.states)
    )
    print(header)
    occupancy = np.zeros((len(sequences), args.states))
    for n, (ident, seq) in enumerate(zip(identifiers, sequences)):
        obs = encode(seq)
        la, ll = model._log_forward(obs)
        lb = model._log_backward(obs)
        gamma = np.exp(la + lb - ll)
        occupancy[n] = gamma.mean(axis=0)
        cells = "".join(f"{100 * v:>10.1f}" for v in occupancy[n])
        print(f"{ident[:19]:<20}{len(seq):>5}{cells}")

    print("\nper-state dominance (sequences with >50% occupancy)")
    for i in range(args.states):
        members = [identifiers[n] for n in range(len(sequences)) if occupancy[n, i] > 0.5]
        print(f"  state {i + 1}: {len(members)}/{len(sequences)} "
              f"{'(' + ', '.join(members) + ')' if members else ''}")
        if len(members) == 1:
            print(f"    WARNING: state {i + 1} is driven by a single sequence; "
                  "the fitted parameters for this state are not supported by "
                  "the seed set as a whole")
    return 0


if __name__ == "__main__":
    sys.exit(main())
