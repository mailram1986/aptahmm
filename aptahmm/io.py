"""Minimal FASTA and plain-text sequence I/O (no external dependencies)."""

from __future__ import annotations


def read_sequences(path):
    """Read sequences from FASTA, or from a plain text file of one per line.

    Blank lines and lines beginning with ``#`` are ignored. Returns
    ``(sequences, identifiers)``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]

    if any(ln.startswith(">") for ln in lines):
        sequences, identifiers = [], []
        current_id, current_seq = None, []
        for ln in lines:
            if ln.startswith(">"):
                if current_id is not None:
                    sequences.append("".join(current_seq))
                    identifiers.append(current_id)
                current_id = ln[1:].strip() or f"seq_{len(sequences) + 1}"
                current_seq = []
            else:
                current_seq.append(ln)
        if current_id is not None:
            sequences.append("".join(current_seq))
            identifiers.append(current_id)
        return sequences, identifiers

    sequences = [ln.split()[0] for ln in lines]
    identifiers = [f"seq_{i + 1}" for i in range(len(sequences))]
    return sequences, identifiers


def write_fasta(path, sequences, identifiers=None, descriptions=None) -> None:
    identifiers = identifiers or [f"seq_{i + 1}" for i in range(len(sequences))]
    with open(path, "w", encoding="utf-8") as fh:
        for i, (ident, seq) in enumerate(zip(identifiers, sequences)):
            desc = f" {descriptions[i]}" if descriptions else ""
            fh.write(f">{ident}{desc}\n{seq}\n")
