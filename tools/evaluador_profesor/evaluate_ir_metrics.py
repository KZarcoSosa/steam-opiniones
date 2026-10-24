#!/usr/bin/env python3
"""Evaluate rankings.tsv against qrels.tsv using P@10, R@10, F1@10, MRR and nDCG@10."""

import argparse
import math
import sys
import pandas as pd

K = 10

def load_files(rankings_path, qrels_path):
    rankings = pd.read_csv(rankings_path, sep="\t",
                           dtype={"query_id": str, "document_id": str})
    qrels = pd.read_csv(qrels_path, sep="\t",
                        dtype={"query_id": str, "document_id": str})
    req_r = {"query_id", "document_id", "rank", "score"}
    req_q = {"query_id", "document_id", "relevance"}
    if req_r - set(rankings.columns):
        raise ValueError(f"rankings.tsv missing columns: {sorted(req_r-set(rankings.columns))}")
    if req_q - set(qrels.columns):
        raise ValueError(f"qrels.tsv missing columns: {sorted(req_q-set(qrels.columns))}")
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="raise").astype(int)
    rankings["score"] = pd.to_numeric(rankings["score"], errors="raise")
    qrels["relevance"] = pd.to_numeric(qrels["relevance"], errors="raise").astype(int)
    return rankings, qrels

def validate(rankings, qrels):
    if not qrels["relevance"].isin([0,1,2]).all():
        raise ValueError("qrels.tsv: relevance must be 0, 1, or 2.")
    if qrels.duplicated(["query_id","document_id"]).any():
        raise ValueError("qrels.tsv contains duplicate query/document judgments.")
    if rankings.duplicated(["query_id","document_id"]).any():
        raise ValueError("rankings.tsv contains duplicate documents within a query.")
    if rankings.duplicated(["query_id","rank"]).any():
        raise ValueError("rankings.tsv contains duplicate ranks within a query.")
    for qid, g in rankings.groupby("query_id"):
        top = g[g["rank"] <= K]
        if len(top) != K or set(top["rank"]) != set(range(1,K+1)):
            raise ValueError(f"{qid}: ranks must be exactly 1 through {K}.")
    missing = set(rankings["query_id"]) - set(qrels["query_id"])
    if missing:
        raise ValueError(f"Queries without qrels: {sorted(missing)}")

def precision_at_k(docs, rel, k=10):
    return sum(rel.get(d,0) > 0 for d in docs[:k]) / k

def recall_at_k(docs, rel, k=10):
    total = sum(r > 0 for r in rel.values())
    return 0.0 if total == 0 else sum(rel.get(d,0) > 0 for d in docs[:k]) / total

def f1_at_k(p, r):
    return 0.0 if p+r == 0 else 2*p*r/(p+r)

def reciprocal_rank(docs, rel, k=10):
    for rank, d in enumerate(docs[:k], 1):
        if rel.get(d,0) > 0:
            return 1.0/rank
    return 0.0

def dcg_at_k(docs, rel, k=10):
    """DCG@k = sum((2^rel_i - 1) / log2(i+1)), exactly as in the example."""
    return sum(((2**rel.get(d,0))-1) / math.log2(i+1)
               for i, d in enumerate(docs[:k], 1))

def ndcg_at_k(docs, rel, k=10):
    dcg = dcg_at_k(docs, rel, k)
    ideal = sorted(rel.values(), reverse=True)[:k]
    ideal += [0] * (k-len(ideal))
    idcg = sum(((2**r)-1) / math.log2(i+1)
               for i, r in enumerate(ideal, 1))
    return 0.0 if idcg == 0 else dcg/idcg

def evaluate(rankings, qrels):
    rows = []
    for qid in sorted(rankings["query_id"].unique()):
        rg = rankings[rankings["query_id"] == qid].sort_values("rank")
        qg = qrels[qrels["query_id"] == qid]
        docs = rg["document_id"].tolist()
        rel = dict(zip(qg["document_id"], qg["relevance"]))
        p = precision_at_k(docs, rel, K)
        r = recall_at_k(docs, rel, K)
        rows.append({
            "query_id": qid,
            "P@10": p,
            "R@10": r,
            "F1@10": f1_at_k(p,r),
            "RR": reciprocal_rank(docs, rel, K),
            "nDCG@10": ndcg_at_k(docs, rel, K)
        })
    df = pd.DataFrame(rows)
    mean = {
        "query_id": "MEAN",
        "P@10": df["P@10"].mean(),
        "R@10": df["R@10"].mean(),
        "F1@10": df["F1@10"].mean(),
        "RR": df["RR"].mean(),  # This mean is MRR.
        "nDCG@10": df["nDCG@10"].mean()
    }
    return df, mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rankings", help="Path to rankings.tsv")
    ap.add_argument("qrels", help="Path to qrels.tsv")
    ap.add_argument("--output", default="evaluation_results.tsv")
    args = ap.parse_args()
    try:
        rankings, qrels = load_files(args.rankings, args.qrels)
        validate(rankings, qrels)
        results, mean = evaluate(rankings, qrels)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nRESULTS BY QUERY")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSYSTEM PERFORMANCE")
    print(f"Mean Precision@10 : {mean['P@10']:.4f}")
    print(f"Mean Recall@10    : {mean['R@10']:.4f}")
    print(f"Mean F1@10        : {mean['F1@10']:.4f}")
    print(f"MRR               : {mean['RR']:.4f}")
    print(f"Mean nDCG@10      : {mean['nDCG@10']:.4f}")

    out = pd.concat([results, pd.DataFrame([mean])], ignore_index=True)
    out.to_csv(args.output, sep="\t", index=False, float_format="%.6f")
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
