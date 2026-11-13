"""Seed-set preprocessing: shared-flank (SELEX primer) detection and trimming.

Aptamer sequences reported in the literature frequently retain the constant
primer-binding regions used during SELEX. Those regions are an artefact of
library construction rather than a product of selection, and because they
are identical across all members of a given library they dominate any
alignment or positional model built from the raw sequences. This module
detects such shared flanks and removes them before model fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _common_suffix_len(a: str, b: str) -> int:
    return _common_prefix_len(a[::-1], b[::-1])


@dataclass
class FlankReport:
    """Shared flanks found in a seed set."""

    prefixes: list = field(default_factory=list)  # (motif, [seq indices])
    suffixes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary_lines(self) -> list:
        lines = []
        for motif, members in self.prefixes:
            lines.append(
                f"5' shared flank ({len(motif)} nt) in {len(members)} sequences: {motif}"
            )
        for motif, members in self.suffixes:
            lines.append(
                f"3' shared flank ({len(motif)} nt) in {len(members)} sequences: {motif}"
            )
        if not lines:
            lines.append("no shared flanks above the detection threshold")
        return lines


def detect_shared_flanks(
    sequences, min_length: int = 12, min_support: int = 2
) -> FlankReport:
    """Find prefixes/suffixes of at least ``min_length`` nt shared by at least
    ``min_support`` sequences.

    Motifs are grown greedily from the longest pairwise common prefix (or
    suffix) and then re-checked against the whole set, so a motif shared by
    more than two sequences is reported once with all its members.

    Because detection is greedy, a reported motif may extend one or two
    nucleotides beyond the true constant region when the adjacent base of the
    variable region happens to coincide in all members. The effect is to trim
    marginally more than the primer itself. Detected motifs are reported in
    full by :meth:`FlankReport.summary_lines` and by the ``inspect``
    subcommand so that they can be checked against the primer sequences given
    in the source publication; pass an explicit :class:`FlankReport` to
    :func:`trim_flanks` to override the automatic choice.
    """
    seqs = [s.upper() for s in sequences]
    n = len(seqs)

    def _collect(common_len_fn, slice_fn):
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                length = common_len_fn(seqs[i], seqs[j])
                if length >= min_length:
                    candidates.append(slice_fn(seqs[i], length))
        # keep the longest distinct motifs, dropping any that is contained in
        # a longer one already accepted
        candidates.sort(key=len, reverse=True)
        accepted = []
        for motif in candidates:
            if any(motif in kept for kept, _ in accepted):
                continue
            members = [k for k in range(n) if slice_fn(seqs[k], len(motif)) == motif]
            if len(members) >= min_support:
                accepted.append((motif, members))
        return accepted

    prefixes = _collect(_common_prefix_len, lambda s, L: s[:L])
    suffixes = _collect(_common_suffix_len, lambda s, L: s[-L:])
    return FlankReport(prefixes=prefixes, suffixes=suffixes)


def trim_flanks(
    sequences,
    report: FlankReport | None = None,
    min_length: int = 12,
    min_support: int = 2,
    min_retained: int = 10,
):
    """Remove detected shared flanks from each sequence.

    A sequence is only trimmed if at least ``min_retained`` nt remain, so a
    short aptamer is never reduced to a stub. Returns the trimmed sequences,
    the report used, and a per-sequence record of what was removed.
    """
    seqs = [s.upper() for s in sequences]
    if report is None:
        report = detect_shared_flanks(seqs, min_length, min_support)

    trimmed, records = [], []
    for i, seq in enumerate(seqs):
        start, end = 0, len(seq)
        removed_5, removed_3 = "", ""
        for motif, members in report.prefixes:
            if i in members and seq.startswith(motif) and start == 0:
                start = len(motif)
                removed_5 = motif
        for motif, members in report.suffixes:
            if i in members and seq.endswith(motif) and end == len(seq):
                end = len(seq) - len(motif)
                removed_3 = motif
        if end - start < min_retained:
            start, end, removed_5, removed_3 = 0, len(seq), "", ""
        trimmed.append(seq[start:end])
        records.append(
            {
                "index": i,
                "original_length": len(seq),
                "trimmed_length": end - start,
                "removed_5_prime": removed_5,
                "removed_3_prime": removed_3,
            }
        )
    return trimmed, report, records


def detect_offset_motifs(
    sequences, motif_length: int = 12, min_support: int = 2, window: int = 30
):
    """Diagnostic scan for shared terminal motifs that are offset from the end.

    :func:`detect_shared_flanks` only finds flanks that begin at position 0
    (or end at the final position). A constant primer region preceded by one
    or two extra transcribed bases is therefore missed, even though it is
    just as much a library artefact. This function scans every k-mer of
    ``motif_length`` within the first and last ``window`` nucleotides and
    reports those occurring in at least ``min_support`` distinct sequences.

    Results are reported for inspection only and are never trimmed
    automatically, because a motif shared by several sequences may also be a
    genuine structural element rather than a primer. Compare the reported
    motifs against the primer sequences in the source publications and, where
    they are primers, pass an explicit :class:`FlankReport` to
    :func:`trim_flanks`.

    Returns a list of ``(motif, end, [sequence indices])`` tuples, longest
    support first, where ``end`` is ``"5-prime"`` or ``"3-prime"``.
    """
    seqs = [s.upper() for s in sequences]
    results = []
    for end, slicer in (
        ("5-prime", lambda s: s[:window]),
        ("3-prime", lambda s: s[-window:]),
    ):
        regions = [slicer(seq) for seq in seqs]
        counts = {}
        for i, region in enumerate(regions):
            for start in range(0, max(0, len(region) - motif_length + 1)):
                counts.setdefault(region[start : start + motif_length], set()).add(i)

        # group k-mers that share the same membership set, then extend each
        # group to the maximal contiguous motif present in all its members
        groups = {}
        for motif, members in counts.items():
            if len(members) >= min_support:
                groups.setdefault(frozenset(members), []).append(motif)

        for members, motifs in groups.items():
            reference = regions[min(members)]
            spans = []
            for motif in motifs:
                idx = reference.find(motif)
                if idx >= 0:
                    spans.append((idx, idx + len(motif)))
            if not spans:
                continue
            lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
            candidate = reference[lo:hi]
            # shrink until every member actually contains the whole motif
            while candidate and not all(candidate in regions[i] for i in members):
                candidate = candidate[:-1]
            if len(candidate) >= motif_length:
                results.append((candidate, end, sorted(members)))

    results.sort(key=lambda t: (-len(t[2]), -len(t[0])))
    deduped = []
    for motif, end, members in results:
        if any(motif in m and end == e for m, e, _ in deduped):
            continue
        deduped.append((motif, end, members))
    return deduped


def gc_content(seq: str) -> float:
    """GC fraction of a sequence."""
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def max_homopolymer_run(seq: str) -> int:
    """Length of the longest single-base run."""
    if not seq:
        return 0
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def seed_statistics(sequences) -> dict:
    """Descriptive statistics for a seed set, for reporting in a methods section."""
    lengths = sorted(len(s) for s in sequences)
    n = len(lengths)
    gcs = [gc_content(s) for s in sequences]
    median = (
        lengths[n // 2] if n % 2 else (lengths[n // 2 - 1] + lengths[n // 2]) / 2.0
    )
    return {
        "n_sequences": n,
        "length_min": lengths[0] if n else 0,
        "length_max": lengths[-1] if n else 0,
        "length_mean": sum(lengths) / n if n else 0.0,
        "length_median": median if n else 0.0,
        "total_nucleotides": sum(lengths),
        "gc_min": min(gcs) if n else 0.0,
        "gc_max": max(gcs) if n else 0.0,
        "gc_mean": sum(gcs) / n if n else 0.0,
    }
