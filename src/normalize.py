"""
Etapa 2a — Normalización del texto (sección 6 de la práctica).

Uso:
    python src/normalize.py

Lee  data/opinions_raw_corpus.tsv
Crea data/opinions_normalized_corpus.tsv

Pasos aplicados a cada reseña (y, por separado, a "nombre del juego + reseña"):
  1. Limpieza: quita etiquetas BBCode de Steam ([h1], [b], [url=...]), URLs,
     caracteres de control y espacios repetidos.
  2. Tokenización con spaCy (modelo es_core_news_sm).
  3. Etiquetado POS y eliminación de las categorías que pide la práctica:
       - artículos           -> DET con PronType=Art   (el, la, un, una...)
       - preposiciones       -> ADP                    (de, en, con, para...)
       - conjunciones        -> CCONJ y SCONJ          (y, o, pero, que, porque...)
       - pronombres          -> PRON y DET posesivos   (yo, se, lo, mi, tu...)
     Como el etiquetador se equivoca con jerga ("fps" -> preposición), un token
     de esas categorías solo se elimina si además es una palabra funcional
     conocida (lista de stopwords de spaCy). Además se quitan signos de
     puntuación, símbolos, espacios y números sueltos.
     NO se quitan adverbios como "no", "muy", "nunca": cambian el sentido de la opinión.
  4. Lematización (token.lemma_), paso a minúsculas y eliminación de acentos
     (optimización = optimizacion; la ñ se conserva).

La función normalize_text() se reutiliza en search.py para normalizar las consultas
EXACTAMENTE igual que el corpus (lo pide la sección 8).
"""
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import spacy

ROOT = Path(__file__).resolve().parent.parent
IN_TSV = ROOT / "data" / "opinions_raw_corpus.tsv"
OUT_TSV = ROOT / "data" / "opinions_normalized_corpus.tsv"

MODEL = "es_core_news_sm"
FUNCTION_POS = {"ADP", "CCONJ", "SCONJ", "PRON"}          # preposiciones, conjunciones, pronombres
NOISE_POS = {"PUNCT", "SYM", "SPACE", "NUM"}
# Palabras funcionales que no vienen en la lista de spaCy
EXTRA_FUNCTION_WORDS = {"cabe", "so", "versus", "vía", "cuyo", "cuya", "cuyos", "cuyas", "alguien", "vos"}

_BBCODE = re.compile(r"\[/?[a-zA-Z0-9*]+(=[^\]]*)?\]")      # [h1] [/b] [url=...] [*]
_URL = re.compile(r"https?://\S+|www\.\S+")
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏  ﻿]")
_SPACES = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1{3,}")                          # "buenooooo" -> "buenoo"

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(MODEL, disable=["ner"])
        except OSError:
            sys.exit(f"Falta el modelo de spaCy. Instálalo con:\n  python -m spacy download {MODEL}")
    return _nlp


def clean_text(text: str) -> str:
    text = _BBCODE.sub(" ", str(text))
    text = _URL.sub(" ", text)
    text = _CTRL.sub(" ", text)
    text = _REPEAT.sub(r"\1\1", text)
    return _SPACES.sub(" ", text).strip()


def is_function_word(tok) -> bool:
    """Las preposiciones, conjunciones, artículos y pronombres son clases cerradas:
    todas están en la lista de palabras funcionales. Si spaCy etiqueta como ADP o PRON
    una palabra que NO está ahí (p. ej. 'fps' -> ADP), es un error del etiquetador
    y la conservamos."""
    w = tok.lower_
    return w in get_nlp().Defaults.stop_words or w in EXTRA_FUNCTION_WORDS


def keep_token(tok) -> bool:
    if tok.pos_ in NOISE_POS:
        return False
    if tok.pos_ in FUNCTION_POS and is_function_word(tok):
        return False
    if tok.pos_ == "DET":
        # Artículos y determinantes posesivos (mi, tu, su...) se eliminan;
        # demostrativos/cuantificadores (este, mucho, ningún) se conservan.
        pron_type = tok.morph.get("PronType")
        if ("Art" in pron_type or "Prs" in pron_type) and is_function_word(tok):
            return False
    if not any(c.isalpha() for c in tok.text):
        return False
    return True


def strip_accents(word: str) -> str:
    """optimización -> optimizacion (conserva la ñ). Los usuarios de Steam
    escriben con y sin acentos; así ambas formas cuentan como el mismo término."""
    word = word.replace("ñ", "\0")
    word = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
    return word.replace("\0", "ñ")


def doc_to_tokens(doc):
    return [strip_accents(tok.lemma_.lower().strip())
            for tok in doc if keep_token(tok) and tok.lemma_.strip()]


def normalize_text(text: str) -> str:
    """Normaliza un solo texto (se usa para las consultas)."""
    return " ".join(doc_to_tokens(get_nlp()(clean_text(text))))


def normalize_many(texts, batch_size=64):
    nlp = get_nlp()
    cleaned = [clean_text(t) for t in texts]
    return [" ".join(doc_to_tokens(d)) for d in nlp.pipe(cleaned, batch_size=batch_size)]


def main():
    if not IN_TSV.exists():
        sys.exit(f"No existe {IN_TSV}. Corre antes: python src/merge_corpus.py")
    df = pd.read_csv(IN_TSV, sep="\t", dtype=str, keep_default_na=False)
    print(f"Normalizando {len(df)} reseñas con spaCy ({MODEL})...")

    df["review_norm"] = normalize_many(df["review"].tolist())
    df["item_review_norm"] = normalize_many((df["item_name"] + ". " + df["review"]).tolist())
    df["n_tokens_raw"] = df["review"].str.split().str.len()
    df["n_tokens_norm"] = df["review_norm"].str.split().str.len()

    cols = ["review_id", "item_id", "item_name", "rating", "date", "category",
            "review_norm", "item_review_norm", "n_tokens_raw", "n_tokens_norm"]
    df[cols].to_csv(OUT_TSV, sep="\t", index=False, encoding="utf-8")

    empty = (df["n_tokens_norm"] == 0).sum()
    print(f"Escrito: {OUT_TSV}")
    print(f"Tokens promedio: {df['n_tokens_raw'].mean():.1f} originales -> {df['n_tokens_norm'].mean():.1f} normalizados")
    if empty:
        print(f"Aviso: {empty} reseñas quedaron vacías tras normalizar")
    ex = df.iloc[0]
    print("\nEjemplo:\n  ORIGINAL:   ", ex["review"][:160], "\n  NORMALIZADA:", ex["review_norm"][:160])


if __name__ == "__main__":
    main()
