"""
Demostración paso a paso de la normalización (para presentar en clase).

Uso:
    python src/demo_normalize.py                 # reseña de ejemplo por defecto
    python src/demo_normalize.py --id 57         # una reseña del corpus por su review_id
    python src/demo_normalize.py --text "Muchos bugs y mala optimización"
    python src/demo_normalize.py --stats         # resumen del corpus normalizado

Muestra, para un texto: 1) limpieza, 2) tokens con su etiqueta POS y si se
quedan o se eliminan (y por qué), 3) lema final, 4) resultado.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import (FUNCTION_POS, NOISE_POS, POSSESSIVES, clean_text, get_nlp,  # noqa: E402
                       is_function_word, keep_token, token_lemma)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "opinions_raw_corpus.tsv"
NORM = ROOT / "data" / "opinions_normalized_corpus.tsv"

POS_ES = {"ADP": "preposición", "CCONJ": "conjunción", "SCONJ": "conj. subordinante",
          "PRON": "pronombre", "DET": "determinante", "PUNCT": "puntuación", "NUM": "número",
          "SYM": "símbolo", "SPACE": "espacio", "NOUN": "sustantivo", "VERB": "verbo",
          "AUX": "auxiliar", "ADJ": "adjetivo", "ADV": "adverbio", "PROPN": "nombre propio",
          "INTJ": "interjección", "X": "otro"}

DEFAULT = ("El juego es muy bonito, pero los jugadores de bajos recursos no lo podrán disfrutar: "
           "tiene muchos bugs y una optimización pésima, ¡apenas llega a 30 fps!")


def reason(tok):
    if tok.pos_ in NOISE_POS:
        return f"se elimina ({POS_ES.get(tok.pos_, tok.pos_)})"
    if tok.pos_ in FUNCTION_POS:
        return (f"se elimina ({POS_ES[tok.pos_]})" if is_function_word(tok)
                else f"SE QUEDA: etiquetado {tok.pos_}, pero no es palabra funcional")
    if tok.pos_ == "DET":
        t = tok.morph.get("PronType")
        if "Art" in t and is_function_word(tok):
            return "se elimina (artículo)"
        if "Prs" in t and is_function_word(tok):
            return "se elimina (posesivo)"
        if tok.lower_ in POSSESSIVES:
            return "se elimina (posesivo)"
    if not any(c.isalpha() for c in tok.text):
        return "se elimina (sin letras)"
    if " " in tok.lemma_.strip():
        return f"se queda, sin el pronombre pegado ({tok.lemma_})"
    return "se queda"


def show(text):
    print("=" * 90)
    print("ORIGINAL:\n ", text)
    cleaned = clean_text(text)
    print("\n1) LIMPIEZA (BBCode, URLs, caracteres de control, espacios):\n ", cleaned)
    doc = get_nlp()(cleaned)
    print("\n2) TOKENIZACIÓN + POS (spaCy es_core_news_sm) + 3) LEMATIZACIÓN:\n")
    print(f"   {'token':16} {'POS':6} {'categoría':20} {'lema':14} decisión")
    print("   " + "-" * 84)
    for tok in doc:
        lemma = token_lemma(tok) if keep_token(tok) else ""
        print(f"   {tok.text:16} {tok.pos_:6} {POS_ES.get(tok.pos_, ''):20} {lemma:14} {reason(tok)}")
    result = [token_lemma(t) for t in doc if keep_token(t)]
    print(f"\n4) RESULTADO ({len(doc)} tokens -> {len(result)} términos):\n ", " ".join(result))
    print("=" * 90)


def stats():
    raw = pd.read_csv(RAW, sep="\t", dtype=str, keep_default_na=False)
    norm = pd.read_csv(NORM, sep="\t", dtype=str, keep_default_na=False)
    a = raw["review"].str.split().str.len()
    b = norm["review_norm"].str.split().str.len()
    vocab_raw = Counter(w.lower() for t in raw["review"] for w in t.split())
    vocab_norm = Counter(w for t in norm["review_norm"] for w in t.split())
    print(f"Reseñas: {len(norm)}")
    print(f"Palabras por reseña:   original {a.mean():.1f}  ->  normalizada {b.mean():.1f}  "
          f"({100 * (1 - b.sum() / a.sum()):.0f} % menos)")
    print(f"Vocabulario distinto:  original {len(vocab_raw):,}  ->  normalizado {len(vocab_norm):,}")
    print("\nTérminos más frecuentes tras normalizar:")
    for w, n in vocab_norm.most_common(20):
        print(f"   {w:15} {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="review_id del corpus")
    ap.add_argument("--text", help="texto libre")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if a.stats:
        return stats()
    if a.id:
        raw = pd.read_csv(RAW, sep="\t", dtype=str, keep_default_na=False).set_index("review_id")
        row = raw.loc[a.id]
        print(f"review_id={a.id} | {row['item_name']} | rating={row['rating']}")
        return show(row["review"])
    show(a.text or DEFAULT)


if __name__ == "__main__":
    main()
