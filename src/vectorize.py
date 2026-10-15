"""
Etapa 2b — Representaciones del Modelo de Espacio Vectorial (sección 7).

Uso:
    python src/vectorize.py

Lee data/opinions_normalized_corpus.tsv y genera 12 representaciones:

    contenido  x  representación  x  características
    review         frequency          unigram  (1,1)
    item_review    binary             bigram   (2,2)
                   tfidf

Cada una se guarda en models/<contenido>_<representacion>_<ngrama>.pkl
con el vectorizador ya entrenado, la matriz documento-término y los review_id
(en el mismo orden que las filas de la matriz).

Por qué así:
  - El texto ya viene normalizado, así que el vectorizador NO vuelve a pasar a
    minúsculas ni quita stopwords: solo separa por espacios (token_pattern=\\S+).
  - frequency = CountVectorizer           (cuántas veces aparece cada término)
    binary    = CountVectorizer(binary=True) (aparece 1 / no aparece 0)
    tfidf     = TfidfVectorizer           (frecuencia ponderada por rareza del término)
"""
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
IN_TSV = ROOT / "data" / "opinions_normalized_corpus.tsv"
MODELS = ROOT / "models"

CONTENTS = {"review": "review_norm", "item_review": "item_review_norm"}
NGRAMS = {"unigram": (1, 1), "bigram": (2, 2)}
REPRESENTATIONS = ("frequency", "binary", "tfidf")


def make_vectorizer(rep: str, ngram_range):
    common = dict(lowercase=False, token_pattern=r"\S+", ngram_range=ngram_range)
    if rep == "frequency":
        return CountVectorizer(**common)
    if rep == "binary":
        return CountVectorizer(binary=True, **common)
    if rep == "tfidf":
        return TfidfVectorizer(**common)
    raise ValueError(rep)


def model_path(content: str, rep: str, ngram: str) -> Path:
    return MODELS / f"{content}_{rep}_{ngram}.pkl"


def main():
    if not IN_TSV.exists():
        sys.exit(f"No existe {IN_TSV}. Corre antes: python src/normalize.py")
    df = pd.read_csv(IN_TSV, sep="\t", dtype=str, keep_default_na=False)
    MODELS.mkdir(exist_ok=True)

    print(f"{'archivo':38} {'docs':>5} {'términos':>9}")
    for content, col in CONTENTS.items():
        for ngram, rng in NGRAMS.items():
            for rep in REPRESENTATIONS:
                vec = make_vectorizer(rep, rng)
                X = vec.fit_transform(df[col])
                path = model_path(content, rep, ngram)
                with path.open("wb") as f:
                    pickle.dump({"vectorizer": vec, "matrix": X,
                                 "review_ids": df["review_id"].tolist()}, f)
                print(f"{path.name:38} {X.shape[0]:>5} {X.shape[1]:>9}")
    print(f"\nModelos en {MODELS}")


if __name__ == "__main__":
    main()
