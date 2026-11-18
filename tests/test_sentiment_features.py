"""Pruebas de la Etapa II (pasos 1 y 2). Las que necesitan spaCy o los léxicos se omiten si faltan."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sentiment_features import (EMOJI_FILE, LIWC_FILE, SEL_FILE, LexiconFeatures, TextPreprocessor,
                                    WORD_RE, strip_accents)

ROOT = Path(__file__).resolve().parents[1]
LABELED = ROOT / "data" / "stage2" / "labeled_documents.tsv"


def pre(text, **kw):
    return TextPreprocessor(**kw).transform([text])[0]


def test_minimal_baseline_only_lowercases_and_tokenizes():
    assert pre("¡Muy BUENO!, no? :)") == "muy bueno no :)"


def test_negation_scope_marks_content_words_and_stops_at_punctuation():
    assert pre("no es nada bueno, es bueno", negation=True) == "no es nada NEG_bueno es bueno"
    assert pre("sin bugs y sin lag", negation=True) == "sin NEG_bugs y sin NEG_lag"


def test_sin_embargo_is_not_negation():
    out = pre("sin embargo me encantó", negation=True)
    assert "NEG_" not in out


def test_emoticon_closes_negation_scope_and_is_kept_as_token():
    out = pre("no me gusta :( es bueno", negation=True)
    assert ":(" in out.split() and "NEG_bueno" not in out


def test_repeats_normalization():
    assert pre("buenooooo muy muy bueno", repeats=True) == "bueno muy bueno"
    assert pre("buenooooo", repeats=False) == "buenooooo"


def test_preprocessor_is_stateless_and_pipeline_compatible():
    from sklearn.base import clone
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    p = Pipeline([("pre", TextPreprocessor(negation=True)),
                  ("vec", TfidfVectorizer(token_pattern=r"\S+", lowercase=False)),
                  ("clf", LogisticRegression())])
    clone(p).set_params(pre__repeats=True)
    X, y = ["muy bueno :)", "no me gusta nada", "excelente", "horrible no lo compres"] * 5, [1, 0, 1, 0] * 5
    p.fit(X, y)
    assert set(p.predict(["excelente :)"])) <= {0, 1}


@pytest.mark.skipif(not all(f.is_file() for f in (SEL_FILE, LIWC_FILE, EMOJI_FILE)),
                    reason="faltan archivos en resources/lexicons/")
def test_lexicon_features_nonnegative_and_expected_columns():
    lf = LexiconFeatures(lemma_fn=lambda t: [strip_accents(w.lower()) for w in WORD_RE.findall(t)])
    X = ["Excelente juego, me encanta :)", "Horrible, no lo recomiendo :(", "sin embargo nunca nadie jugó"]
    M = lf.transform(X)
    assert M.shape == (3, len(lf.get_feature_names_out())) and (M >= 0).all()
    names = list(lf.get_feature_names_out())
    assert M[2, names.index("n_negators")] == 2          # nunca, nadie; "sin embargo" no cuenta
    assert M[1, names.index("emoji_neg")] > 0


spacy_ok = importlib.util.find_spec("spacy") is not None and importlib.util.find_spec("es_core_news_sm") is not None


@pytest.mark.skipif(not spacy_ok, reason="spaCy/es_core_news_sm no instalado")
def test_negators_survive_function_word_removal_when_protected():
    out = pre("Sin bugs y no lo recomiendo", lemmatize=True, remove_function_words=True).split()
    assert "sin" in out and "no" in out


@pytest.mark.skipif(not LABELED.is_file(), reason="primero: python main.py labels")
def test_labeled_documents_splits_are_disjoint_complete_and_stratified():
    d = pd.read_csv(LABELED, sep="\t", dtype=str, keep_default_na=False)
    assert d.document_id.is_unique and set(d.split) == {"train", "dev", "test"}
    assert list(d.columns[:6]) == ["document_id", "review", "text", "rating", "label", "split"]
    assert (d.text == d.review).all()
    counts = d.split.value_counts()
    assert abs(counts["train"] / len(d) - 0.70) < 0.01 and abs(counts["dev"] - counts["test"]) <= 1
    share = pd.crosstab(d.split, d.label, normalize="index")["positive"]
    assert share.max() - share.min() < 0.03                      # estratificación
    assert set(d.label) == {"positive", "negative"}