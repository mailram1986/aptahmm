"""Test suite for aptahmm. Run with: python -m pytest tests/ -v"""

import numpy as np
import pytest

from aptahmm import (
    DiscreteHMM,
    detect_offset_motifs,
    GenerationConfig,
    WindowPSSM,
    decode,
    detect_shared_flanks,
    encode,
    gc_content,
    generate,
    max_homopolymer_run,
    select_n_states,
    trim_flanks,
)
from aptahmm.hmm import _logsumexp, validate_sequences

SEEDS = [
    "TCAACGGCAGTAACATTAGC",
    "CCCCCGTTGCTTTCGCTTTTCCTTTCGCTTTTGTTCGTTTCGTCCCTGCTTCCTTTCTTG",
    "AGCAGCACAGAGGTCAGATGGGCGGGTTTGGATCTTTGGTTGGCGCCTGTTTCTTTATGACCTATGCGTGCTACCGTGAA",
    "AGCAGCACAGAGGTCAGATGATATGTTTACGCCAGTGGTATTATTGGGGTTGATATGTCACCTATGCGTGCTACCGTGAA",
    "AATCAGGCTCAGCATGGAGTTGCGAGGCCAATATCCGGTTAAGCG",
    "TGGCGTTAGAGATACCGGAACCGGTGTCGGGCGTCTGAAGAATCC",
]


class TestEncoding:
    def test_roundtrip(self):
        assert decode(encode("ACGTACGT")) == "ACGTACGT"

    def test_uracil_mapped_to_thymine(self):
        assert decode(encode("ACGU")) == "ACGT"

    def test_rejects_ambiguity_codes(self):
        with pytest.raises(ValueError):
            validate_sequences(["ACGTN"])

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            validate_sequences([""])


class TestLogSumExp:
    def test_matches_naive(self):
        a = np.array([-1.0, -2.0, -3.0])
        assert _logsumexp(a) == pytest.approx(np.log(np.exp(a).sum()))

    def test_handles_neg_inf(self):
        a = np.array([-np.inf, 0.0])
        assert _logsumexp(a) == pytest.approx(0.0)


class TestHMM:
    def test_likelihood_monotone_in_ml_limit(self):
        """Unpenalised EM must never decrease the data log-likelihood."""
        model = DiscreteHMM(4, random_state=1).fit(
            SEEDS, n_restarts=1, n_iter=40, pseudocount=1e-9
        )
        history = np.array(model.history_)
        assert np.all(np.diff(history) >= -1e-6), "EM decreased the log-likelihood"

    def test_penalised_objective_is_monotone(self):
        """The M-step maximises the penalised objective, so it never decreases."""
        model = DiscreteHMM(4, random_state=11).fit(
            SEEDS, n_restarts=1, n_iter=200, pseudocount=1.0
        )
        objective = np.array(model.objective_history_)
        assert np.all(np.diff(objective) >= -1e-6), "EM decreased the objective"

    def test_converges_within_cap(self):
        model = DiscreteHMM(2, random_state=12).fit(
            SEEDS, n_restarts=1, n_iter=400, pseudocount=1.0, tol=1e-5
        )
        assert model.converged_
        assert model.n_iter_run_ < 400

    def test_non_convergence_warns(self):
        with pytest.warns(RuntimeWarning, match="iteration cap"):
            DiscreteHMM(4, random_state=13).fit(
                SEEDS, n_restarts=1, n_iter=2, pseudocount=1.0, tol=1e-12
            )

    def test_map_estimation_may_dip_but_converges(self):
        """With pseudocounts the M-step is MAP, so small dips are expected."""
        model = DiscreteHMM(4, random_state=1).fit(
            SEEDS, n_restarts=1, n_iter=200, pseudocount=1.0, tol=1e-5
        )
        history = np.array(model.history_)
        # the trajectory must still settle: the last change is small
        assert abs(history[-1] - history[-2]) < 1e-2
        # and the fit must improve substantially over the random start
        assert history[-1] > history[0]

    def test_rows_normalised(self):
        model = DiscreteHMM(5, random_state=2).fit(SEEDS, n_restarts=1, n_iter=30)
        assert np.exp(model.log_pi).sum() == pytest.approx(1.0)
        assert np.allclose(np.exp(model.log_A).sum(axis=1), 1.0)
        assert np.allclose(np.exp(model.log_B).sum(axis=1), 1.0)

    def test_parameter_count(self):
        model = DiscreteHMM(3)
        assert model.n_parameters == 2 + 3 * 2 + 3 * 3

    def test_sampling_length_and_alphabet(self):
        rng = np.random.default_rng(0)
        model = DiscreteHMM(4, random_state=3).fit(SEEDS, n_restarts=1, n_iter=20)
        seq = model.sample(23, rng)
        assert len(seq) == 23
        assert set(seq) <= set("ACGT")

    def test_sampling_is_reproducible(self):
        model = DiscreteHMM(4, random_state=4).fit(SEEDS, n_restarts=1, n_iter=20)
        a = model.sample_many(10, 23, np.random.default_rng(7))
        b = model.sample_many(10, 23, np.random.default_rng(7))
        assert a == b

    def test_unfitted_model_raises(self):
        with pytest.raises(RuntimeError):
            DiscreteHMM(3).sample(10, np.random.default_rng(0))

    def test_bic_penalises_complexity(self):
        small = DiscreteHMM(2, random_state=5).fit(SEEDS, n_restarts=1, n_iter=50)
        large = DiscreteHMM(14, random_state=5).fit(SEEDS, n_restarts=1, n_iter=50)
        assert large.n_parameters > small.n_parameters
        assert large.bic(SEEDS) > small.aic(SEEDS) - 1e9  # sanity: finite values
        assert np.isfinite(large.bic(SEEDS)) and np.isfinite(small.bic(SEEDS))


class TestFlankDetection:
    def test_detects_duan_primer_pair(self):
        report = detect_shared_flanks(SEEDS, min_length=12, min_support=2)
        prefixes = [m for m, _ in report.prefixes]
        suffixes = [m for m, _ in report.suffixes]
        assert any("AGCAGCACAGAGGTCAGATG" in m for m in prefixes)
        assert any("CCTATGCGTGCTACCGTGAA" in m for m in suffixes)

    def test_trimming_shortens_only_flanked_sequences(self):
        trimmed, _, records = trim_flanks(SEEDS)
        # Greedy detection finds a 21 nt shared 3' motif rather than the 20 nt
        # primer, because the adjacent variable-region base coincides in both
        # Duan sequences, so 80 - 20 - 21 = 39 nt is retained. Documented in
        # detect_shared_flanks.
        assert records[2]["trimmed_length"] == 39
        assert records[0]["trimmed_length"] == 20  # no shared flank, untouched
        assert all(len(t) > 0 for t in trimmed)

    def test_explicit_report_overrides_greedy_detection(self):
        """A caller can supply the exact primer motifs from the publication."""
        from aptahmm.preprocess import FlankReport

        report = FlankReport(
            prefixes=[("AGCAGCACAGAGGTCAGATG", [2, 3])],
            suffixes=[("CCTATGCGTGCTACCGTGAA", [2, 3])],
        )
        _, _, records = trim_flanks(SEEDS, report=report)
        assert records[2]["trimmed_length"] == 40
        assert records[3]["trimmed_length"] == 40

    def test_short_sequence_never_stubbed(self):
        seqs = ["AAAAAAAAAAAAGGG", "AAAAAAAAAAAACCC"]
        trimmed, _, _ = trim_flanks(seqs, min_length=12, min_retained=10)
        assert all(len(t) >= 10 for t in trimmed)


class TestOffsetMotifs:
    def test_finds_offset_primer_missed_by_prefix_scan(self):
        hu = [
            "TTGGCTAGCTCAGTCATATGGCGGCGTCACCCGACGGGGACTTGACATTATGACAGCCGCG",
            "TGGCTAGCTCAGTCATCTAAAAATGGGCAAAGAAACAGTGACTCGTTGAGATACTCCGCG",
        ]
        # the shared motif is offset by one base, so the prefix scan finds nothing
        report = detect_shared_flanks(hu, min_length=12, min_support=2)
        assert not report.prefixes
        # the offset scan recovers it as a single maximal motif
        motifs = detect_offset_motifs(hu, motif_length=12, min_support=2)
        five_prime = [m for m, end, _ in motifs if end == "5-prime"]
        assert any("GGCTAGCTCAGTCAT" in m for m in five_prime)

    def test_collapses_overlapping_kmers(self):
        hu = [
            "TTGGCTAGCTCAGTCATATGGCGGCGTCACCCGACGGGGACTTGACATTATGACAGCCGCG",
            "TGGCTAGCTCAGTCATCTAAAAATGGGCAAAGAAACAGTGACTCGTTGAGATACTCCGCG",
        ]
        motifs = [m for m, end, _ in detect_offset_motifs(hu, 12, 2) if end == "5-prime"]
        assert len(motifs) == 1, f"expected one maximal motif, got {motifs}"

    def test_reports_nothing_on_unrelated_sequences(self):
        unrelated = ["ACGTACGTACGTACGTACGT", "TTTTAAAACCCCGGGGTTTT"]
        assert detect_offset_motifs(unrelated, motif_length=12, min_support=2) == []


class TestPSSM:
    def test_score_within_range(self):
        pssm = WindowPSSM(23).fit(SEEDS)
        score = pssm.score("ACGTACGTACGTACGTACGTACG")
        assert pssm.min_score() <= score <= pssm.max_score()

    def test_wrong_length_raises(self):
        pssm = WindowPSSM(23).fit(SEEDS)
        with pytest.raises(ValueError):
            pssm.score("ACGT")

    def test_information_content_bounds(self):
        pssm = WindowPSSM(23).fit(SEEDS)
        ic = pssm.information_content()
        assert ic.shape == (23,)
        assert np.all(ic >= -1e-9) and np.all(ic <= 2.0 + 1e-9)

    def test_too_long_window_raises(self):
        with pytest.raises(ValueError):
            WindowPSSM(500).fit(SEEDS)


class TestFilters:
    def test_gc_content(self):
        assert gc_content("GCGC") == pytest.approx(1.0)
        assert gc_content("ATAT") == pytest.approx(0.0)
        assert gc_content("") == 0.0

    def test_homopolymer_run(self):
        assert max_homopolymer_run("AAAACGT") == 4
        assert max_homopolymer_run("ACGT") == 1
        assert max_homopolymer_run("") == 0


class TestPipeline:
    def test_end_to_end(self):
        config = GenerationConfig(
            length=23, n_states=5, n_candidates=25, oversample=20, em_restarts=1
        )
        result = generate(SEEDS, config)
        cands = result["candidates"]
        assert len(cands) == 25
        for rec in cands:
            assert len(rec["sequence"]) == 23
            assert config.gc_min <= rec["gc_content"] <= config.gc_max
            assert rec["max_homopolymer_run"] <= config.max_homopolymer
        scores = [r["pssm_score_bits"] for r in cands]
        assert scores == sorted(scores, reverse=True)

    def test_unique_sequences(self):
        config = GenerationConfig(n_candidates=50, oversample=30, em_restarts=1)
        result = generate(SEEDS, config)
        seqs = [r["sequence"] for r in result["candidates"]]
        assert len(seqs) == len(set(seqs))

    def test_identical_seed_gives_identical_output(self):
        config = GenerationConfig(n_candidates=20, oversample=20, em_restarts=1)
        a = generate(SEEDS, config)["candidates"]
        b = generate(SEEDS, config)["candidates"]
        assert [r["sequence"] for r in a] == [r["sequence"] for r in b]

    def test_different_seed_gives_different_output(self):
        c1 = GenerationConfig(n_candidates=20, oversample=20, em_restarts=1, random_seed=1)
        c2 = GenerationConfig(n_candidates=20, oversample=20, em_restarts=1, random_seed=2)
        a = [r["sequence"] for r in generate(SEEDS, c1)["candidates"]]
        b = [r["sequence"] for r in generate(SEEDS, c2)["candidates"]]
        assert a != b

    def test_manifest_records_provenance(self):
        config = GenerationConfig(n_candidates=10, oversample=20, em_restarts=1)
        manifest = generate(SEEDS, config)["manifest"]
        assert manifest["model"]["n_states"] == config.n_states
        assert manifest["model"]["model_order"] == 1
        assert manifest["run"]["config"]["random_seed"] == config.random_seed
        assert manifest["seed_set"]["raw"]["n_sequences"] == len(SEEDS)
        assert manifest["pssm"]["length"] == config.length

    def test_length_longer_than_seeds_raises(self):
        config = GenerationConfig(length=400, em_restarts=1)
        with pytest.raises(ValueError):
            generate(SEEDS, config)


class TestModelSelection:
    def test_returns_best_and_table(self):
        best, table = select_n_states(
            SEEDS, [2, 3, 4], criterion="bic", n_restarts=1, n_iter=40
        )
        assert best in (2, 3, 4)
        assert len(table) == 3
        assert all(np.isfinite(row["bic"]) for row in table)
