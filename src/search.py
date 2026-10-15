"""
Etapa 3 — Recuperación de información (sección 8).

Uso:
    python src/search.py "se traba mucho y tiene mala optimización"
    python src/search.py "historia aburrida" --content item_review --features bigram --rep binary

Para cada consulta:
  1. Se normaliza EXACTAMENTE igual que el corpus (normalize.normalize_text).
  2. Se extraen las características indicadas (unigramas o bigramas).
  3. Se representa en el mismo espacio vectorial (vectorizer.transform).
  4. Se calcula la similitud coseno contra todos los documentos.
  5. Se ordena de mayor a menor y se muestran los 10 más similares con su score.
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import normalize_text  # noqa: E402
from vectorize import model_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_TSV = ROOT / "data" / "opinions_raw_corpus.tsv"

_cache = {}


def load_model(content, rep, features):
    key = (content, rep, features)
    if key not in _cache:
        path = model_path(content, rep, features)
        if not path.exists():
            sys.exit(f"No existe {path}. Corre antes: python src/vectorize.py")
        with path.open("rb") as f:
            _cache[key] = pickle.load(f)
    return _cache[key]


def search(query, content="review", features="unigram", rep="tfidf", k=10):
    """Regresa [(review_id, score), ...] ordenado de mayor a menor."""
    model = load_model(content, rep, features)
    q_norm = normalize_text(query)
    q_vec = model["vectorizer"].transform([q_norm])
    if q_vec.nnz == 0:
        return q_norm, []   # ningún término de la consulta existe en el vocabulario
    scores = cosine_similarity(q_vec, model["matrix"]).ravel()
    order = scores.argsort()[::-1][:k]
    return q_norm, [(model["review_ids"][i], float(scores[i])) for i in order if scores[i] > 0]


def main():
    ap = argparse.ArgumentParser(description="Top-10 reseñas más similares a una consulta.")
    ap.add_argument("query")
    ap.add_argument("--content", choices=["review", "item_review"], default="review",
                    help="review = solo texto; item_review = nombre del juego + texto")
    ap.add_argument("--features", choices=["unigram", "bigram"], default="unigram")
    ap.add_argument("--rep", choices=["frequency", "binary", "tfidf"], default="tfidf")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    q_norm, results = search(args.query, args.content, args.features, args.rep, args.k)
    raw = pd.read_csv(RAW_TSV, sep="\t", dtype=str, keep_default_na=False).set_index("review_id")

    print(f'Consulta: "{args.query}"')
    print(f"Normalizada: {q_norm}")
    print(f"Contenido={args.content} | características={args.features} | representación={args.rep}\n")
    if not results:
        print("Sin resultados: ningún término de la consulta aparece en el corpus con esta configuración.")
        return
    for rank, (rid, score) in enumerate(results, 1):
        row = raw.loc[rid]
        rating = "[+]" if row["rating"] == "1" else "[-]"
        text = row["review"][:180] + ("…" if len(row["review"]) > 180 else "")
        print(f"{rank:2}. [{score:.4f}] id={rid} {rating} {row['item_name']}\n    {text}\n")


if __name__ == "__main__":
    main()
