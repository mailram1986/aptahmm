"""End-to-end generation pipeline: seeds -> HMM -> candidates -> filters."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np

from . import __version__
from .hmm import DiscreteHMM, validate_sequences
from .preprocess import (
    detect_shared_flanks,
    gc_content,
    max_homopolymer_run,
    seed_statistics,
    trim_flanks,
)
from .pssm import WindowPSSM


@dataclass
class GenerationConfig:
    """Every parameter that affects the output, recorded in the manifest."""

    length: int = 23
    n_states: int = 6
    n_candidates: int = 500
    oversample: int = 40
    random_seed: int = 20260101
    pseudocount: float = 1.0
    em_iterations: int = 300
    em_tolerance: float = 1e-5
    em_restarts: int = 5
    trim_flanks: bool = True
    flank_min_length: int = 12
    flank_min_support: int = 2
    gc_min: float = 0.40
    gc_max: float = 0.65
    max_homopolymer: int = 4
    pssm_percentile: float = 50.0
    pssm_pseudocount: float = 1.0
    deduplicate: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def select_n_states(
    sequences, candidate_states, criterion: str = "bic", random_state: int = 0, **fit_kw
):
    """Fit one model per candidate state count and rank them by BIC or AIC.

    Returns ``(best_n_states, table)`` where ``table`` is a list of dicts
    suitable for writing to CSV and quoting in a methods section.
    """
    sequences = validate_sequences(sequences)
    table = []
    for k in candidate_states:
        model = DiscreteHMM(n_states=k, random_state=random_state + k).fit(
            sequences, **fit_kw
        )
        table.append(
            {
                "n_states": k,
                "n_free_parameters": model.n_parameters,
                "log_likelihood": model.log_likelihood_,
                "aic": model.aic(sequences),
                "bic": model.bic(sequences),
                "em_iterations": model.n_iter_run_,
                "converged": model.converged_,
            }
        )
    key = "bic" if criterion.lower() == "bic" else "aic"
    best = min(table, key=lambda row: row[key])
    return best["n_states"], table


def passes_composition_filters(seq: str, config: GenerationConfig) -> tuple:
    """Return ``(ok, reason)`` for the sequence-composition filters."""
    gc = gc_content(seq)
    if gc < config.gc_min:
        return False, f"gc_below_{config.gc_min:.2f}"
    if gc > config.gc_max:
        return False, f"gc_above_{config.gc_max:.2f}"
    run = max_homopolymer_run(seq)
    if run > config.max_homopolymer:
        return False, f"homopolymer_run_{run}"
    return True, "pass"


def generate(seed_sequences, config: GenerationConfig | None = None, verbose=False):
    """Run the full pipeline and return a result dictionary.

    Keys: ``candidates`` (list of dicts, highest PSSM score first),
    ``rejected`` (list of dicts), ``manifest`` (full provenance record),
    ``model`` (the fitted :class:`DiscreteHMM`), ``pssm``, ``model_selection``.
    """
    config = config or GenerationConfig()
    raw = validate_sequences(seed_sequences)
    rng = np.random.default_rng(config.random_seed)

    # ---- 1. preprocessing -------------------------------------------------
    flank_report = detect_shared_flanks(
        raw, config.flank_min_length, config.flank_min_support
    )
    if config.trim_flanks:
        training, flank_report, trim_records = trim_flanks(
            raw,
            report=flank_report,
            min_length=config.flank_min_length,
            min_support=config.flank_min_support,
        )
    else:
        training = list(raw)
        trim_records = []

    usable = [s for s in training if len(s) >= config.length]
    if not usable:
        raise ValueError(
            f"no seed sequence is at least {config.length} nt after trimming; "
            "lower --length or disable trimming"
        )

    if verbose:
        for line in flank_report.summary_lines():
            print(f"[preprocess] {line}")
        print(f"[preprocess] {len(usable)}/{len(training)} sequences usable")

    # ---- 2. model fitting -------------------------------------------------
    model = DiscreteHMM(config.n_states, random_state=config.random_seed).fit(
        training,
        n_iter=config.em_iterations,
        tol=config.em_tolerance,
        pseudocount=config.pseudocount,
        n_restarts=config.em_restarts,
        verbose=verbose,
    )
    if verbose:
        print(
            f"[hmm] {config.n_states} states, ll={model.log_likelihood_:.3f}, "
            f"{model.n_iter_run_} EM iterations, converged={model.converged_}"
        )

    # ---- 3. PSSM ----------------------------------------------------------
    pssm = WindowPSSM(config.length, pseudocount=config.pssm_pseudocount).fit(usable)

    # ---- 4. sampling and filtering ---------------------------------------
    target = config.n_candidates
    pool_size = max(target * config.oversample, target)
    seen = set()
    scored, rejected = [], []

    drawn = 0
    while drawn < pool_size and len(scored) < pool_size:
        seq = model.sample(config.length, rng)
        drawn += 1
        if config.deduplicate and seq in seen:
            rejected.append({"sequence": seq, "reason": "duplicate"})
            continue
        seen.add(seq)
        ok, reason = passes_composition_filters(seq, config)
        record = {
            "sequence": seq,
            "length": len(seq),
            "gc_content": round(gc_content(seq), 4),
            "max_homopolymer_run": max_homopolymer_run(seq),
            "pssm_score_bits": round(pssm.score(seq), 4),
            "hmm_log_likelihood": round(model.log_likelihood([seq]), 4),
        }
        if ok:
            scored.append(record)
        else:
            record["reason"] = reason
            rejected.append(record)

    if not scored:
        raise RuntimeError(
            "no sampled sequence passed the composition filters; widen the GC "
            "window or raise --max-homopolymer"
        )

    # PSSM percentile threshold, applied after composition filtering
    all_scores = np.array([r["pssm_score_bits"] for r in scored])
    threshold = float(np.percentile(all_scores, config.pssm_percentile))
    kept = [r for r in scored if r["pssm_score_bits"] >= threshold]
    for r in scored:
        if r["pssm_score_bits"] < threshold:
            rec = dict(r)
            rec["reason"] = "pssm_below_threshold"
            rejected.append(rec)

    kept.sort(key=lambda r: r["pssm_score_bits"], reverse=True)
    kept = kept[:target]
    for rank, record in enumerate(kept, start=1):
        record["rank"] = rank
        record["id"] = f"aptahmm_{rank:04d}"

    if verbose:
        print(
            f"[sample] drew {drawn}, {len(scored)} passed composition, "
            f"{len(kept)} returned (PSSM >= {threshold:.3f} bits)"
        )

    # ---- 5. manifest ------------------------------------------------------
    manifest = {
        "software": {
            "name": "aptahmm",
            "version": __version__,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": config.as_dict(),
            "sequences_drawn": drawn,
            "pssm_threshold_bits": round(threshold, 4),
            "n_returned": len(kept),
        },
        "seed_set": {
            "raw": seed_statistics(raw),
            "after_trimming": seed_statistics(training),
            "usable_for_pssm": len(usable),
            "shared_flanks": flank_report.as_dict(),
            "shared_flank_summary": flank_report.summary_lines(),
            "trim_records": trim_records,
        },
        "model": model.to_dict(),
        "pssm": pssm.to_dict(),
    }

    return {
        "candidates": kept,
        "rejected": rejected,
        "manifest": manifest,
        "model": model,
        "pssm": pssm,
    }


def write_outputs(result: dict, outdir) -> dict:
    """Write FASTA, CSV and JSON outputs; return the paths written."""
    import csv
    import os

    os.makedirs(outdir, exist_ok=True)
    paths = {}

    fasta = os.path.join(outdir, "candidates.fasta")
    with open(fasta, "w", encoding="utf-8") as fh:
        for rec in result["candidates"]:
            fh.write(
                f">{rec['id']} pssm_bits={rec['pssm_score_bits']} "
                f"gc={rec['gc_content']}\n{rec['sequence']}\n"
            )
    paths["fasta"] = fasta

    csv_path = os.path.join(outdir, "candidates.csv")
    fields = [
        "rank",
        "id",
        "sequence",
        "length",
        "gc_content",
        "max_homopolymer_run",
        "pssm_score_bits",
        "hmm_log_likelihood",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in result["candidates"]:
            writer.writerow({k: rec[k] for k in fields})
    paths["csv"] = csv_path

    manifest_path = os.path.join(outdir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(result["manifest"], fh, indent=2)
    paths["manifest"] = manifest_path

    return paths
