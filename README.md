# aptahmm

**A hidden Markov model based DNA aptamer sequence generator.**

Version 1.0.0 · MIT licence · Python ≥ 3.10 · numpy only

---

## What this software does

`aptahmm` fits a first-order discrete hidden Markov model (HMM) to a curated
set of experimentally validated single-stranded DNA (ssDNA) aptamers and
samples fixed-length candidate sequences from the fitted model. Sampled
candidates are ranked by a position-specific scoring matrix (PSSM) derived
from the same seed set and filtered on GC content and homopolymer run length.

The package also provides the preprocessing and diagnostic steps that a seed
set of this kind requires: detection and removal of the constant SELEX
primer-binding regions that published aptamer sequences frequently retain,
information-criterion comparison across hidden-state counts including a
composition-only baseline, EM convergence monitoring on the correct
objective, and per-sequence hidden-state occupancies.

## What this software does not do

**The model does not encode, predict or imply affinity for any molecular
target.** It is trained on aptamers raised against unrelated targets and
learns only the general sequence composition of SELEX-derived ssDNA
aptamers. Target specificity must be established downstream — by docking,
molecular dynamics and ultimately by experiment. Output sequences are
candidate starting points for such screening, not predicted binders.

This distinction is not a formality. A generator of this kind is a sampling
prior over sequence space; any statement about what its output binds is a
claim about the downstream stages, not about the HMM.

**The model generates sequence only. It does not compute, predict or imply
secondary structure, folding free energy or any other thermodynamic
property.** The HMM is a first-order model over nucleotide composition and
adjacent-base transitions; it carries no representation of base pairing,
stem–loop topology or duplex stability. No quantity in the output is an
energy: `pssm_score_bits` is a positional log-odds score in bits and
`hmm_log_likelihood` is the log-probability of the sequence under the fitted
model, and neither is expressed in kcal/mol or relates monotonically to
folding stability. A candidate ranked highly here may fold poorly, or not
at all. The GC window applied during filtering (0.40–0.65) is a composition
constraint chosen for synthesis and folding tractability, not a structural
calculation. Structure must therefore be assessed in a separate downstream
step — by passing the generated sequences to a folding tool (`mfold`,
`RNAfold`, `UNAFold`) or equivalent — before any structure-based selection
or ranking.

## Installation

```bash
git clone <repository-url>
cd aptahmm
python3 -m venv --system-site-packages .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python3 -m pip install -e .
# or, without installing the package:
python3 -m pip install -r requirements.txt
```

The virtual environment is not optional on current Linux distributions.
Debian, Ubuntu and Fedora ship Python as an externally managed environment
(PEP 668), and `pip install` into the system interpreter is refused there with
an `error: externally-managed-environment` message. `--system-site-packages`
lets the environment reuse a system `numpy` if one is already present instead
of building its own copy.

Run the tests:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest tests/ -q      # 37 tests
```

`requirements.txt` is `numpy` alone; `pytest` is needed only for the test
suite and for step 6 of `scripts/reproduce.sh`. Omitting it does not affect
the five steps that generate the reported results.

## Quick start

```bash
# 1. inspect the seed set: statistics and shared primer regions
python3 -m aptahmm.cli inspect --seeds data/seeds_ssdna_aptamers.fasta

# 2. choose the number of hidden states by BIC (include the 1-state baseline)
python3 -m aptahmm.cli select-states --seeds data/seeds_ssdna_aptamers.fasta \
    --min-states 1 --max-states 6 --em-iterations 500 --em-tolerance 1e-6

# 3. generate candidates
python3 -m aptahmm.cli generate --seeds data/seeds_ssdna_aptamers.fasta \
    --length 23 --states 2 -n 200 --out results/run_k2
```

Every result quoted below is reproduced by `scripts/reproduce.sh`.

Python API:

```python
from aptahmm import GenerationConfig, generate, write_outputs
from aptahmm.io import read_sequences

seeds, ids = read_sequences("data/seeds_ssdna_aptamers.fasta")
config = GenerationConfig(length=23, n_states=2, n_candidates=200, random_seed=20260101)
result = generate(seeds, config)
write_outputs(result, "results/run_k2")
```

## Method

### 1. Seed-set preprocessing

Published aptamer sequences commonly retain the constant primer-binding
regions used to amplify the SELEX library. These regions are an artefact of
library construction rather than a product of selection, and because they are
identical across all members of a library they dominate any positional model
built from the raw sequences. `aptahmm` detects them in two ways:

- **Terminal flanks** (`detect_shared_flanks`) — prefixes or suffixes of at
  least `--flank-min-length` nt shared by at least `--flank-min-support`
  sequences. These are removed before fitting.
- **Offset motifs** (`detect_offset_motifs`) — shared motifs that sit near a
  terminus but do not begin exactly at it, which the prefix scan misses.
  These are **reported only, never trimmed automatically**, because such a
  motif may also be a genuine structural element. Check them against the
  primers given in the source publication and, where they are primers, supply
  an explicit `FlankReport` to `trim_flanks`.

Detection is greedy, so a reported terminal flank may extend one or two
nucleotides past the true constant region when the adjacent variable-region
base happens to coincide in all members. The detected motifs are printed in
full so this can be checked.

### 2. Model

A first-order discrete HMM over the alphabet {A, C, G, T}:

- **Model order:** 1 (the next hidden state depends only on the current one).
- **Hidden states:** `--states`, chosen by information criterion.
- **Free parameters:** (K−1) initial + K(K−1) transition + 3K emission.
- **Emissions:** a categorical distribution over the four bases per state.

Parameters are estimated by Baum–Welch expectation–maximisation computed in
log space, with multiple random restarts (`--em-restarts`) to reduce the risk
of settling in a poor local optimum — a real risk with a small training set.
Symmetric Dirichlet pseudocounts (`--pseudocount`, default 1.0) are added at
every M-step so that no parameter is driven to exactly zero by a sparse
training set.

**Convergence is monitored on the penalised objective**, log-likelihood plus
log prior, because that is the quantity the M-step maximises when
pseudocounts are used. Monitoring the bare log-likelihood instead is a
mistake that is easy to make and that prevents convergence from ever being
detected: with a pseudocount of 1.0 the likelihood drifts slowly downward
near the optimum as the estimate is pulled toward the prior, so a
likelihood-based criterion runs to the iteration cap. On the seed set below,
switching to the correct objective reduced the run from 300 iterations
(not converged) to 37 (converged). A `RuntimeWarning` is raised if the cap is
reached, and `objective_history_` records the trajectory for inspection.

### 3. Candidate sampling and filtering

Sequences of the requested length are drawn from the fitted model and
filtered in order:

| Filter | Default | Rationale |
|---|---|---|
| Duplicate removal | on | distinct candidates only |
| GC content | 0.40–0.65 | synthesis and folding tractability |
| Max homopolymer run | 4 | synthesis fidelity, avoids artefactual tracts |
| PSSM log-odds | top 50% | positional base composition of the seed set |

The pipeline over-samples (`--oversample`, default 40×) so that the requested
number of candidates survives filtering. Candidates are returned ranked by
PSSM score.

### 4. The PSSM

The seed sequences differ greatly in length and are not mutually alignable in
any biologically meaningful way, so a conventional profile PSSM over
alignment columns is not defined for them. The matrix here is built instead
from every sliding window of the target length across the trimmed seed set,
scored as log-odds in bits against the seed set's own base composition. It
therefore encodes the positional base composition of an *L*-nucleotide window
of aptamer sequence space and is used to rank candidates. **It does not
assert homology**, and its per-position information content should be
inspected (`pssm.information_content()`) before any structural interpretation
is placed on it.

## Reproducibility

Every run writes a `run_manifest.json` recording the full configuration, the
RNG seed, the fitted transition and emission matrices, the PSSM and its
threshold, seed-set statistics before and after trimming, the detected
flanks, and the Python, numpy and platform versions. A fixed `--random-seed`
reproduces an identical candidate list; this is asserted in the test suite.

## Results on the bundled seed set

`data/seeds_ssdna_aptamers.fasta` contains 12 ssDNA aptamers from 11 published
studies (`data/seed_references.csv`). **All were raised against targets
unrelated to any target of interest here.** They are included to characterise
general ssDNA aptamer sequence composition and to serve as a worked example.

### Seed-set composition

| | n | length range | mean | median | GC range |
|---|---|---|---|---|---|
| Raw | 12 | 20–90 nt | 60.3 | 60.5 | 45.0–59.0% |
| After flank trimming | 12 | 20–90 nt | 53.5 | 52.5 | 38.5–59.0% |

Detected flanks:

- 5′ 20 nt `AGCAGCACAGAGGTCAGATG` in 2 sequences (Duan 2023 pair) — trimmed
- 3′ 21 nt `ACCTATGCGTGCTACCGTGAA` in 2 sequences (Duan 2023 pair) — trimmed
  (the canonical primer is the 20-nt `CCTATGCGTGCTACCGTGAA`; one further base
  coincides)
- 5′ 16 nt `TGGCTAGCTCAGTCAT` in 2 sequences (Hu 2021 pair) — **reported, not
  trimmed**: the motif is offset by one nucleotide from the terminus, so it is
  not a common prefix

Trimming removes 82 of 724 nt (11%), all of it from the two Duan sequences,
which fall from 80 to 39 nt each.

### Hidden-state selection

Produced by `scripts/reproduce.sh` step 2 (5 restarts, 500 iterations,
tolerance 1e-6, pseudocount 1.0, seed 20260101):

| States | Params | log L | AIC | BIC | EM iters | converged |
|---:|---:|---:|---:|---:|---:|:--|
| **1** | **3** | -883.418 | 1772.84 | **1786.23** | 3 | yes |
| 2 | 9 | -870.525 | **1759.05** | 1799.23 | 253 | yes |
| 3 | 17 | -878.520 | 1791.04 | 1866.94 | 500 | **no** |
| 4 | 27 | -877.524 | 1809.05 | 1929.59 | 344 | yes |
| 5 | 39 | -883.425 | 1844.85 | 2018.97 | 258 | yes |
| 6 | 53 | -883.427 | 1872.85 | 2109.48 | 189 | yes |

**The two criteria disagree decisively and in opposite directions.** BIC
selects a single hidden state, by 13.0 units over the two-state model; AIC
selects two states, by 13.8 units over one. Both margins are past the
conventional threshold of roughly 10 units, so this is not a borderline case
that more careful fitting would resolve - it is the signature of a model
order the data do not determine. BIC's heavier penalty (`params x log 642`
against AIC's `2 x params`) dominates at this sample size, and with 642
nucleotides of training data that penalty is the more appropriate one. A
single-state HMM has no hidden structure at all: it is a base-composition
sampler. The empirical comparison two sections below settles which criterion
to believe here.

Two cautions on reading this table. The three-state model did not converge
within 500 iterations, so its figures bound its fit rather than report a
fitted optimum. And because EM finds local optima, the log-likelihood at a
given state count varies with `--em-restarts`: an earlier run with 3 restarts
reached -874.56 for two states rather than -870.52, moving the BIC margin
from 13.0 to 21.1 units. The direction of every comparison held across those
runs, but quote figures from a stated command rather than treating them as
properties of the data alone.

**Interpretation: this seed set does not support hidden-state structure.**
Twelve heterogeneous sequences totalling 642 nt after trimming are too few,
and too unlike one another, to identify transition and emission matrices
beyond overall base composition. This is a property of the data, not of the
implementation.

### Why the two-state model is unstable

Fitting two states anyway and examining per-sequence posterior occupancy
(`scripts/state_occupancy.py`) shows the second state is driven by one
sequence:

| Seed | A% | T% | occupancy of the T-rich state |
|---|---:|---:|---:|
| Schmitz2023 | 0.0 | 50.0 | **97.8%** |
| Duan2023a | 5.1 | 43.6 | 76.2% |
| Effah2023 | 23.6 | 27.8 | 19.2% |
| Rizzotto2023 | 15.5 | 31.0 | 10.5% |
| *the remaining eight* | — | — | **< 8% each** |

One state models an A-free, T-rich composition that is essentially the
Schmitz 2023 aptamer alone. (Hidden states are not identifiable up to
permutation, so the index attached to this state varies between runs — the
diagnostic output identifies it by its emission profile, not its number.) Removing that single sequence and refitting
changes the emission matrices substantially, confirming the instability. A
state supported by one of twelve training sequences should not be given a
mechanistic interpretation, and `state_occupancy.py` prints an explicit
warning when it detects one. Note that with 12 sequences the state is
dominated by two seeds (Schmitz2023 and, partly, Duan2023a) while the other
ten sit in the generic state at >89% occupancy.

### Generated candidates

With `--states 2 --length 23 -n 200`, 8 000 sequences were drawn, 5 264 passed
the composition filters, and the top 200 by PSSM score were returned
(threshold −0.03 bits). The five highest-scoring:

| Rank | Sequence | PSSM (bits) | GC% |
|---:|---|---:|---:|
| 1 | `ACCCCAACAAGGTGGCTTGGTGT` | 1.11 | 56.5 |
| 2 | `CCACAACCGCGGGTTTCATTTGG` | 1.09 | 56.5 |
| 3 | `CAAAAGGCAAGATGTTATCCTCG` | 1.07 | 43.5 |
| 4 | `ACAAGTAACTCGGAGTTGGTTCC` | 1.04 | 47.8 |
| 5 | `AAACAAGACGGCTTTGTCTTCGC` | 1.04 | 47.8 |

### The hidden states make no measurable difference

Running the same pipeline with the BIC-selected single-state model
(`results/run_k1/`) and comparing the top 200 candidates:

| Model | n | mean PSSM (bits) | median | max |
|---|---:|---:|---:|---:|
| 1 state (composition only) | 200 | 0.702 | 0.668 | **1.181** |
| 2 states | 200 | 0.708 | 0.678 | 1.113 |

The difference in mean score is **+0.007 bits** in favour of the two-state
model — negligible, and the single-state model produces the single
highest-scoring candidate. Both drew 8 000 sequences for 200 returned.

This is the direct empirical counterpart of the BIC result: on this seed set
the hidden states change neither the candidates nor their scores, so the HMM
is operating as a base-composition sampler. Any methodological claim for the
hidden-state layer has to be supported on a seed set where this comparison
comes out differently.

## Recommendations for use

Reporting the above honestly is more useful than concealing it, and it points
to what would strengthen a study of this kind:

1. **Report the 1-state baseline.** Any claim that an HMM contributes
   something must be measured against a composition-only sampler. If the two
   produce comparable candidates, the hidden states are not earning their
   place.
2. **Scale the seed set.** Identifying even a handful of hidden states needs
   hundreds of sequences, ideally from a single SELEX library so the constant
   regions are known and the variable regions are the same length.
3. **Trim primers, and say so.** Report the primer sequences removed and the
   lengths before and after. An untrimmed model's most confident parameters
   describe a PCR primer.
4. **Validate the generator, not just the candidates.** The defensible claim
   for a sampling prior is enrichment: the fraction of generated sequences
   that fold stably, or that survive downstream filtering, relative to random
   sequences of the same length and GC content. That comparison is cheap and
   it is the measurement that makes a generator a method rather than a
   utility.
5. **Keep target specificity downstream.** Nothing in this model is
   target-aware.

## Repository contents

```
aptahmm/
  __init__.py        package API
  hmm.py             discrete HMM: log-space Baum-Welch, BIC/AIC, sampling
  preprocess.py      primer-flank detection and trimming, composition stats
  pssm.py            sliding-window position-specific scoring matrix
  generate.py        end-to-end pipeline, run manifest, output writers
  io.py              FASTA and plain-text sequence I/O
  cli.py             command-line interface (inspect, select-states, generate)
data/
  seeds_ssdna_aptamers.fasta   12 curated ssDNA aptamer seeds
  seed_references.csv          source reference and notes for each seed
scripts/
  reproduce.sh                 reproduces every result in this README
  state_occupancy.py           per-sequence hidden-state occupancy diagnostic
tests/
  test_aptahmm.py              37 tests
results/                       outputs written by scripts/reproduce.sh
```

## Output files

| File | Contents |
|---|---|
| `candidates.fasta` | ranked candidates, PSSM score and GC in the header |
| `candidates.csv` | rank, id, sequence, length, GC, max homopolymer run, PSSM score, HMM log-likelihood |
| `run_manifest.json` | full configuration, fitted matrices, PSSM, seed statistics, detected flanks, software versions |

## Limitations

- The model is first-order and position-independent; it captures base
  composition and dinucleotide-level transitions, not secondary structure.
  Folding free energy is not computed — pair the output with a folding tool
  (`mfold`, `RNAfold`, `UNAFold`) before structural selection.
- Offset primer motifs are reported but not trimmed automatically, and so
  require manual checking against the source publications.
- EM finds local optima. The restart count mitigates this; it does not
  eliminate it. Compare `objective_history_` across restarts before quoting
  fitted parameters.
- No target information is used anywhere in the pipeline.

## How to cite

See `CITATION.cff`. Replace the placeholder repository URL and DOI before
publishing the deposit.

## Licence

MIT. See `LICENSE`.
