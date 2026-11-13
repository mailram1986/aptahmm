"""Command-line interface for aptahmm."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from . import __version__
from .generate import GenerationConfig, generate, select_n_states, write_outputs
from .io import read_sequences
from .preprocess import (
    detect_offset_motifs,
    detect_shared_flanks,
    seed_statistics,
    trim_flanks,
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seeds", required=True, help="FASTA or one-sequence-per-line text file"
    )
    parser.add_argument(
        "--no-trim",
        dest="trim_flanks",
        action="store_false",
        help="keep shared SELEX primer flanks instead of trimming them",
    )
    parser.add_argument(
        "--flank-min-length",
        type=int,
        default=12,
        help="minimum shared flank length to treat as a primer region (default 12)",
    )
    parser.add_argument(
        "--flank-min-support",
        type=int,
        default=2,
        help="minimum number of sequences sharing a flank (default 2)",
    )
    parser.set_defaults(trim_flanks=True)


def cmd_inspect(args) -> int:
    """Report seed-set statistics and detected primer flanks."""
    sequences, identifiers = read_sequences(args.seeds)
    report = detect_shared_flanks(
        sequences, args.flank_min_length, args.flank_min_support
    )
    trimmed, _, records = trim_flanks(
        sequences,
        report=report,
        min_length=args.flank_min_length,
        min_support=args.flank_min_support,
    )

    raw_stats = seed_statistics(sequences)
    trim_stats = seed_statistics(trimmed)

    print(f"aptahmm {__version__} - seed set inspection\n")
    print(f"file: {args.seeds}")
    print(f"sequences: {raw_stats['n_sequences']}\n")

    print("shared terminal flanks (trimmed)")
    for line in report.summary_lines():
        print(f"  {line}")
    print()

    offset = detect_offset_motifs(
        sequences, motif_length=args.flank_min_length, min_support=args.flank_min_support
    )
    not_trimmed = [
        (m, e, mem)
        for m, e, mem in offset
        if not any(m in t for t, _ in report.prefixes + report.suffixes)
    ]
    print("offset terminal motifs (reported only, NOT trimmed)")
    if not_trimmed:
        for motif, end, members in not_trimmed[:10]:
            names = ", ".join(identifiers[i] for i in members)
            print(f"  {end} {len(motif)} nt in {len(members)}: {motif}  [{names}]")
        print(
            "  -> check these against the primers in the source publications;\n"
            "     if any is a primer region, supply an explicit FlankReport."
        )
    else:
        print("  none above threshold")
    print()

    print(f"{'id':<28}{'raw':>6}{'trim':>6}  removed")
    for ident, rec in zip(identifiers, records):
        removed = []
        if rec["removed_5_prime"]:
            removed.append(f"5'-{len(rec['removed_5_prime'])}nt")
        if rec["removed_3_prime"]:
            removed.append(f"3'-{len(rec['removed_3_prime'])}nt")
        print(
            f"{ident[:27]:<28}{rec['original_length']:>6}{rec['trimmed_length']:>6}"
            f"  {', '.join(removed) if removed else '-'}"
        )
    print()

    for label, stats in (("raw", raw_stats), ("trimmed", trim_stats)):
        print(
            f"{label:<8} n={stats['n_sequences']:<3} "
            f"length {stats['length_min']}-{stats['length_max']} "
            f"(mean {stats['length_mean']:.1f}, median {stats['length_median']:.1f})  "
            f"GC {100 * stats['gc_min']:.1f}-{100 * stats['gc_max']:.1f}% "
            f"(mean {100 * stats['gc_mean']:.1f}%)"
        )
    return 0


def cmd_select(args) -> int:
    """Rank candidate hidden-state counts by BIC and AIC."""
    sequences, _ = read_sequences(args.seeds)
    if args.trim_flanks:
        sequences, _, _ = trim_flanks(
            sequences,
            min_length=args.flank_min_length,
            min_support=args.flank_min_support,
        )

    states = list(range(args.min_states, args.max_states + 1))
    best, table = select_n_states(
        sequences,
        states,
        criterion=args.criterion,
        random_state=args.random_seed,
        n_iter=args.em_iterations,
        tol=args.em_tolerance,
        pseudocount=args.pseudocount,
        n_restarts=args.em_restarts,
    )

    print(f"aptahmm {__version__} - hidden-state model selection\n")
    print(f"{'states':>7}{'params':>8}{'logL':>14}{'AIC':>12}{'BIC':>12}{'iters':>7}")
    for row in table:
        print(
            f"{row['n_states']:>7}{row['n_free_parameters']:>8}"
            f"{row['log_likelihood']:>14.3f}{row['aic']:>12.2f}"
            f"{row['bic']:>12.2f}{row['em_iterations']:>7}"
        )
    print(f"\nbest by {args.criterion.upper()}: {best} hidden states")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)
        print(f"written: {args.out}")
    return 0


def cmd_generate(args) -> int:
    """Fit the model and write candidate sequences."""
    sequences, _ = read_sequences(args.seeds)
    config = GenerationConfig(
        length=args.length,
        n_states=args.states,
        n_candidates=args.n,
        oversample=args.oversample,
        random_seed=args.random_seed,
        pseudocount=args.pseudocount,
        em_iterations=args.em_iterations,
        em_tolerance=args.em_tolerance,
        em_restarts=args.em_restarts,
        trim_flanks=args.trim_flanks,
        flank_min_length=args.flank_min_length,
        flank_min_support=args.flank_min_support,
        gc_min=args.gc_min,
        gc_max=args.gc_max,
        max_homopolymer=args.max_homopolymer,
        pssm_percentile=args.pssm_percentile,
        pssm_pseudocount=args.pssm_pseudocount,
        deduplicate=not args.allow_duplicates,
    )

    result = generate(sequences, config, verbose=args.verbose)
    paths = write_outputs(result, args.out)

    print(f"\naptahmm {__version__} - generated {len(result['candidates'])} candidates")
    print(f"  {paths['fasta']}")
    print(f"  {paths['csv']}")
    print(f"  {paths['manifest']}")
    print("\ntop 10 by PSSM score:")
    print(f"  {'rank':>4}  {'sequence':<26}{'bits':>8}{'GC%':>7}")
    for rec in result["candidates"][:10]:
        print(
            f"  {rec['rank']:>4}  {rec['sequence']:<26}"
            f"{rec['pssm_score_bits']:>8.2f}{100 * rec['gc_content']:>7.1f}"
        )
    if args.print_manifest:
        print()
        print(json.dumps(result["manifest"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aptahmm",
        description=(
            "Hidden Markov model based DNA aptamer sequence generator. The model "
            "captures general sequence features of SELEX-derived ssDNA aptamers; "
            "it does not predict affinity for any target."
        ),
    )
    parser.add_argument("--version", action="version", version=f"aptahmm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser(
        "inspect", help="report seed-set statistics and shared primer flanks"
    )
    _add_common(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    p_select = sub.add_parser(
        "select-states", help="rank hidden-state counts by BIC/AIC"
    )
    _add_common(p_select)
    p_select.add_argument(
        "--min-states",
        type=int,
        default=1,
        help="smallest state count to test; 1 is the composition-only baseline",
    )
    p_select.add_argument("--max-states", type=int, default=12)
    p_select.add_argument("--criterion", choices=["bic", "aic"], default="bic")
    p_select.add_argument("--pseudocount", type=float, default=1.0)
    p_select.add_argument("--em-iterations", type=int, default=300)
    p_select.add_argument("--em-tolerance", type=float, default=1e-5)
    p_select.add_argument("--em-restarts", type=int, default=5)
    p_select.add_argument("--random-seed", type=int, default=20260101)
    p_select.add_argument("--out", default=None, help="write the table to CSV")
    p_select.set_defaults(func=cmd_select)

    p_gen = sub.add_parser("generate", help="fit the model and sample candidates")
    _add_common(p_gen)
    p_gen.add_argument("--length", type=int, default=23, help="candidate length (nt)")
    p_gen.add_argument("--states", type=int, default=6, help="hidden states")
    p_gen.add_argument("-n", type=int, default=500, help="candidates to return")
    p_gen.add_argument("--oversample", type=int, default=40)
    p_gen.add_argument("--pseudocount", type=float, default=1.0)
    p_gen.add_argument("--em-iterations", type=int, default=300)
    p_gen.add_argument("--em-tolerance", type=float, default=1e-5)
    p_gen.add_argument("--em-restarts", type=int, default=5)
    p_gen.add_argument("--gc-min", type=float, default=0.40)
    p_gen.add_argument("--gc-max", type=float, default=0.65)
    p_gen.add_argument("--max-homopolymer", type=int, default=4)
    p_gen.add_argument("--pssm-percentile", type=float, default=50.0)
    p_gen.add_argument("--pssm-pseudocount", type=float, default=1.0)
    p_gen.add_argument("--allow-duplicates", action="store_true")
    p_gen.add_argument("--random-seed", type=int, default=20260101)
    p_gen.add_argument("--out", default="aptahmm_output")
    p_gen.add_argument("--verbose", action="store_true")
    p_gen.add_argument("--print-manifest", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
