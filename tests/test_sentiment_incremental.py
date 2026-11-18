"""Pruebas de la cadena incremental y de la estadística pareada (no requieren spaCy)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sentiment_incremental import build_chain, paired_stats, verdict, without_factor


def test_chain_is_cumulative_and_ordered():
    chain, skipped = build_chain(["terraria"], core_only=True, have_spacy=False, have_speller=False)
    assert [s["id"] for s in chain] == ["S0_base", "S1_repeticiones", "S2_negacion", "S3_lexicos"]
    assert chain[0]["preproc"] == {} and chain[0]["lex"] == []
    for prev, cur in zip(chain, chain[1:]):                    # cada paso conserva todo lo anterior
        assert set(prev["preproc"].items()) <= set(cur["preproc"].items())
        assert set(prev["lex"]) <= set(cur["lex"])
    assert chain[-1]["preproc"] == {"repeats": True, "negation": True}
    assert chain[-1]["lex"] == ["sel", "liwc", "emoji", "negation"]


def test_steps_needing_missing_resources_are_skipped_not_silently_included():
    chain, skipped = build_chain([], core_only=False, have_spacy=False, have_speller=False)
    assert {s for s, _ in skipped} == {"S4_lemas", "S5_ortografia", "S6_funcionales"}
    assert all("lemmatize" not in s["preproc"] for s in chain)


def test_without_factor_removes_only_that_factor():
    chain, _ = build_chain([], core_only=True, have_spacy=False, have_speller=False)
    full = chain[-1]
    pre, lex = without_factor(full, chain, "negación")
    assert pre == {"repeats": True} and lex == full["lex"]
    pre, lex = without_factor(full, chain, "léxicos")
    assert pre == full["preproc"] and lex == []
    pre, lex = without_factor(full, chain, "repeticiones")
    assert pre == {"negation": True}


def test_paired_stats_identical_scores_give_no_effect():
    a = np.array([.7, .72, .68, .75, .71, .69])
    m, lo, hi, p, w, k = paired_stats(a, a, 0.25)
    assert m == 0 and p == 1.0 and w == 0 and k == 6


def test_paired_stats_matches_corrected_ttest_formula():
    from scipy import stats
    a = np.array([.70, .71, .69, .72, .70, .68, .73, .71])
    b = a + np.array([.02, .01, .03, -.01, .02, .00, .04, .01])
    d = b - a
    k, ratio = len(d), 0.25
    t = d.mean() / np.sqrt((1 / k + ratio) * d.var(ddof=1))
    expected_p = 2 * (1 - stats.t.cdf(abs(t), k - 1))
    m, lo, hi, p, w, kk = paired_stats(a, b, ratio)
    assert abs(m - d.mean()) < 1e-12 and abs(p - expected_p) < 1e-9 and lo < m < hi and w == 6 and kk == 8


def test_verdict_labels():
    assert verdict(0.001, 0.9) == "sin cambio apreciable"
    assert verdict(0.03, 0.01).startswith("MEJORA")
    assert verdict(-0.03, 0.01).startswith("EMPEORA")
    assert "no concluyente" in verdict(0.03, 0.4)


spacy_ok = importlib.util.find_spec("spacy") is not None and importlib.util.find_spec("es_core_news_sm") is not None


@pytest.mark.skipif(not spacy_ok, reason="spaCy/es_core_news_sm no instalado")
def test_full_chain_with_spacy_includes_lemmas_and_function_words():
    chain, skipped = build_chain([], core_only=False, have_spacy=True, have_speller=False)
    ids = [s["id"] for s in chain]
    assert "S4_lemas" in ids and "S6_funcionales" in ids
    assert chain[-1]["preproc"]["lemmatize"] and chain[-1]["preproc"]["remove_function_words"]