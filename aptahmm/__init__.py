"""aptahmm - hidden Markov model based DNA aptamer sequence generator.

The package fits a first-order discrete HMM to a curated set of
experimentally validated ssDNA aptamers and samples fixed-length candidate
sequences from the fitted model. Candidates are then ranked by a
position-specific scoring matrix derived from the same seed set and filtered
on GC content and homopolymer runs.

The model encodes the general sequence characteristics of SELEX-derived
ssDNA aptamers. It does not encode affinity for any particular target, and
no claim of target specificity should be made on the basis of its output;
target selection is the role of the downstream docking and simulation
stages of an aptamer design pipeline.
"""

__version__ = "1.0.0"

from .hmm import ALPHABET, DiscreteHMM, decode, encode  # noqa: F401
from .preprocess import (  # noqa: F401
    detect_offset_motifs,
    detect_shared_flanks,
    gc_content,
    max_homopolymer_run,
    seed_statistics,
    trim_flanks,
)
from .pssm import WindowPSSM  # noqa: F401
from .generate import (  # noqa: F401
    GenerationConfig,
    generate,
    select_n_states,
    write_outputs,
)

__all__ = [
    "__version__",
    "ALPHABET",
    "DiscreteHMM",
    "WindowPSSM",
    "GenerationConfig",
    "generate",
    "select_n_states",
    "write_outputs",
    "detect_shared_flanks",
    "detect_offset_motifs",
    "trim_flanks",
    "seed_statistics",
    "gc_content",
    "max_homopolymer_run",
    "encode",
    "decode",
]
