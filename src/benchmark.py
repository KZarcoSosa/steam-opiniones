"""
Etapa 4 — Benchmark y evaluación (sección 9).

Uso:
    python src/benchmark.py                                   # config por defecto (review, unigram, tfidf)
    python src/benchmark.py --content review --features bigram --rep binary
    python src/benchmark.py --all                             # las 12 configuraciones + tabla comparativa

Necesita (hechos A MANO por el equipo, con el corpus congelado):
    benchmark/queries.tsv   query_id  difficulty_level  query
    benchmark/qrels.tsv     query_id  document_id  relevance      (1 o 2; los 0 no se ponen)

Genera:
    results/rankings.tsv                 query_id document_id rank score   (top-10 por consulta)
    results/metrics_per_query.tsv        P@10 R@10 F1@10 RR nDCG@10 por consulta
    results/metrics_summary.tsv          promedios (total y por nivel de dificultad)
    results/comparison_all_configs.tsv   (solo con --all)

Métricas (todas sobre el top-10):
    P@10   = relevantes recuperados / 10
    R@10   = relevantes recuperados / total de relevantes de la consulta en qrels
    F1@10  = 2·P·R / (P+R)
    RR     = 1 / posición del primer relevante (0 si no hay ninguno);  MRR = promedio
    nDCG@10 = DCG / DCG_ideal, con DCG = Σ rel_i / log2(i+1)  (rel = 0, 1 o 2)
"""
import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import search  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"
RESULTS = ROOT / "results"
K = 10
METRICS = ["P@10", "R@10", "F1@10", "RR", "nDCG@10"]


def load_benchmark():
    q_path, r_path = BENCH / "queries.tsv", BENCH / "qrels.tsv"
    for p in (q_path, r_path):
        if not p.exists():
            sys.exit(f"Falta {p}. Revisa benchmark/LEEME.md")
    queries = pd.read_csv(q_path, sep="\t", dtype={"query_id": str, "difficulty_level": int, "query": str})
    qrels = pd.read_csv(r_path, sep="\t", dtype={"query_id": str, "document_id": str, "relevance": int})
    qrels = qrels[qrels["relevance"] > 0]
    rel = {qid: dict(zip(g["document_id"], g["relevance"])) for qid, g in qrels.groupby("query_id")}
    return queries, rel


def evaluate_query(ranked_ids, relevant: dict):
    top = ranked_ids[:K]
    hits = [d for d in top if d in relevant]
    p = len(hits) / K
    r = len(hits) / len(relevant) if relevant else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    rr = next((1 / (i + 1) for i, d in enumerate(top) if d in relevant), 0.0)
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(top))
    ideal = sorted(relevant.values(), reverse=True)[:K]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg else 0.0
    return {"P@10": p, "R@10": r, "F1@10": f1, "RR": rr, "nDCG@10": ndcg}


def run(content, features, rep, queries, rel, write=True):
    rank_rows, metric_rows = [], []
    for q in queries.itertuples():
        _, results = search(q.query, content, features, rep, K)
        ranked = [rid for rid, _ in results]
        for i, (rid, score) in enumerate(results, 1):
            rank_rows.append({"query_id": q.query_id, "document_id": rid, "rank": i, "score": round(score, 4)})
        m = evaluate_query(ranked, rel.get(q.query_id, {}))
        metric_rows.append({"query_id": q.query_id, "difficulty_level": q.difficulty_level,
                            "query": q.query, "n_relevant": len(rel.get(q.query_id, {})),
                            **{k: round(v, 4) for k, v in m.items()}})
    per_query = pd.DataFrame(metric_rows)
    summary = per_query.groupby("difficulty_level")[METRICS].mean()
    summary.loc["promedio"] = per_query[METRICS].mean()
    summary = summary.rename(columns={"RR": "MRR"}).round(4)

    if write:
        RESULTS.mkdir(exist_ok=True)
        pd.DataFrame(rank_rows).to_csv(RESULTS / "rankings.tsv", sep="\t", index=False)
        per_query.to_csv(RESULTS / "metrics_per_query.tsv", sep="\t", index=False)
        summary.to_csv(RESULTS / "metrics_summary.tsv", sep="\t")
    return per_query, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", choices=["review", "item_review"], default="review")
    ap.add_argument("--features", choices=["unigram", "bigram"], default="unigram")
    ap.add_argument("--rep", choices=["frequency", "binary", "tfidf"], default="tfidf")
    ap.add_argument("--all", action="store_true", help="Evalúa las 12 configuraciones")
    args = ap.parse_args()
    queries, rel = load_benchmark()
    pd.set_option("display.width", 160)

    if args.all:
        rows = []
        for content in ("review", "item_review"):
            for features in ("unigram", "bigram"):
                for rep in ("frequency", "binary", "tfidf"):
                    _, s = run(content, features, rep, queries, rel, write=False)
                    rows.append({"content": content, "features": features, "rep": rep,
                                 **s.loc["promedio"].to_dict()})
        comp = pd.DataFrame(rows).sort_values("nDCG@10", ascending=False)
        RESULTS.mkdir(exist_ok=True)
        comp.to_csv(RESULTS / "comparison_all_configs.tsv", sep="\t", index=False)
        print(comp.to_string(index=False))
        best = comp.iloc[0]
        print(f"\nMejor configuración: {best.content} / {best.features} / {best.rep}. "
              f"Generando rankings.tsv con ella...")
        args.content, args.features, args.rep = best.content, best.features, best.rep

    per_query, summary = run(args.content, args.features, args.rep, queries, rel)
    print(f"\nConfiguración: {args.content} / {args.features} / {args.rep}\n")
    print(per_query.drop(columns="query").to_string(index=False))
    print("\nPromedios (1=fácil, 2=media, 3=difícil):\n")
    print(summary.to_string())
    print(f"\nArchivos en {RESULTS}")


if __name__ == "__main__":
    main()
