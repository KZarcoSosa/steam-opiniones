"""
Ayudante para ARMAR el benchmark (no es parte de la evaluación).

Uso:
    python src/label_pool.py

Para cada consulta de benchmark/queries.tsv junta candidatos de las 12
configuraciones (top-20 de cada una, técnica de "pooling") y los escribe en
benchmark/pool/<query_id>.tsv con una columna `relevance` vacía.

El equipo lee cada reseña y pone 2 (muy relevante), 1 (parcial) o 0 (no relevante).
OJO: para las consultas difíciles el sistema casi no encuentra nada (es justo lo
que mide la práctica), así que también hay que buscar relevantes A MANO en
data/opinions_raw_corpus.tsv (Ctrl+F con sinónimos) y agregarlas.

Luego `python src/label_pool.py --collect` junta todo lo marcado con 1 o 2 en
benchmark/qrels.tsv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import search  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"
POOL = BENCH / "pool"
RAW_TSV = ROOT / "data" / "opinions_raw_corpus.tsv"


def build_pools():
    queries = pd.read_csv(BENCH / "queries.tsv", sep="\t", dtype=str)
    raw = pd.read_csv(RAW_TSV, sep="\t", dtype=str, keep_default_na=False).set_index("review_id")
    POOL.mkdir(parents=True, exist_ok=True)
    for q in queries.itertuples():
        out = POOL / f"{q.query_id}.tsv"
        if out.exists():
            print(f"{out.name} ya existe, no se sobrescribe (bórralo si quieres regenerarlo)")
            continue
        votes = {}
        for content in ("review", "item_review"):
            for features in ("unigram", "bigram"):
                for rep in ("frequency", "binary", "tfidf"):
                    for rid, _ in search(q.query, content, features, rep, 20)[1]:
                        votes[rid] = votes.get(rid, 0) + 1
        rows = [{"query_id": q.query_id, "document_id": rid, "relevance": "",
                 "veces_recuperado": n, "item_name": raw.loc[rid, "item_name"],
                 "review": raw.loc[rid, "review"]}
                for rid, n in sorted(votes.items(), key=lambda x: -x[1])]
        pd.DataFrame(rows).to_csv(out, sep="\t", index=False, encoding="utf-8")
        print(f"{out.name}: {len(rows)} candidatos para '{q.query}'")


def collect():
    frames = [pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False) for p in sorted(POOL.glob("Q*.tsv"))]
    df = pd.concat(frames)
    df = df[df["relevance"].isin(["1", "2"])][["query_id", "document_id", "relevance"]]
    df = df.drop_duplicates(["query_id", "document_id"])
    df.to_csv(BENCH / "qrels.tsv", sep="\t", index=False)
    counts = df.groupby("query_id").size()
    print(counts.to_string())
    bad = counts[(counts < 10) | (counts > 15)]
    if len(bad):
        print(f"\nAviso: la práctica pide 10-15 relevantes por consulta. Revisar: {list(bad.index)}")
    print(f"\nEscrito {BENCH / 'qrels.tsv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="Junta las etiquetas en benchmark/qrels.tsv")
    collect() if ap.parse_args().collect else build_pools()
