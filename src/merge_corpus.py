"""
Une los JSONL de todos los integrantes en data/opinions_raw_corpus.tsv
y reporta la distribución (por juego, rating, fecha, fuente) que pide la práctica.

Uso:
    python src/merge_corpus.py
    python src/merge_corpus.py --max-per-game 45 --seed 7

IMPORTANTE: review_id es un entero secuencial (es el document_id de qrels.tsv).
Una vez que empiecen a etiquetar el benchmark, NO vuelvan a correr este script
con datos nuevos: congelen el TSV (commit + tag) para que los IDs no cambien.
"""
import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_TSV = ROOT / "data" / "opinions_raw_corpus.tsv"
OUT_STATS = ROOT / "data" / "corpus_stats.txt"

COLUMNS = ["review_id", "item_id", "item_name", "review", "rating", "date",
           "category", "source", "url", "source_id", "language", "playtime_hours",
           "votes_up", "collected_by", "retrieved_at"]


def tsv_safe(text: str) -> str:
    """Solo lo mínimo para que el TSV no se rompa (tabs/saltos de línea).
    La normalización real (tokenizar, lematizar, etc.) va en la etapa 2."""
    return re.sub(r"[\t\r\n]+", " ", text).strip()


def load_records():
    records = []
    for path in sorted(RAW_DIR.glob("*/*.jsonl")):
        member = path.parent.name
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rec["collected_by"] = member
                    records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-game", type=int, default=None,
                    help="Recorta cada juego a N reseñas (muestreo aleatorio reproducible)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_records()
    if not records:
        raise SystemExit("No hay archivos en data/raw/*/*.jsonl")

    # Deduplicar por id de Steam (por si dos personas bajaron el mismo juego)
    by_id = {}
    for r in records:
        by_id.setdefault(r["recommendationid"], r)
    dups = len(records) - len(by_id)
    records = list(by_id.values())

    if args.max_per_game:
        rng = random.Random(args.seed)
        grouped = {}
        for r in records:
            grouped.setdefault(r["appid"], []).append(r)
        records = []
        for appid in sorted(grouped):
            g = sorted(grouped[appid], key=lambda x: x["recommendationid"])
            records += rng.sample(g, min(args.max_per_game, len(g)))

    # Orden determinista -> IDs reproducibles
    records.sort(key=lambda r: (int(r["appid"]), int(r["recommendationid"])))

    rows = []
    for i, r in enumerate(records, start=1):
        ts = r.get("timestamp_created")
        date = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat() if ts else ""
        pt = r.get("playtime_at_review_min")
        rows.append({
            "review_id": i,
            "item_id": r["appid"],
            "item_name": r["item_name"],
            "review": tsv_safe(r["review"]),
            "rating": 1 if r.get("voted_up") else 0,   # Steam: 1 = Recomendado, 0 = No recomendado
            "date": date,
            "category": "videojuegos|" + ",".join(r.get("genres", [])),
            "source": "Steam",
            "url": f"https://store.steampowered.com/app/{r['appid']}/",
            "source_id": r["recommendationid"],
            "language": r.get("language", ""),
            "playtime_hours": round(pt / 60, 1) if pt is not None else "",
            "votes_up": r.get("votes_up", ""),
            "collected_by": r["collected_by"],
            "retrieved_at": r.get("retrieved_at", ""),
        })

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(OUT_TSV, sep="\t", index=False, encoding="utf-8")

    lines = [
        f"Total de reseñas: {len(df)}  (duplicados eliminados: {dups})",
        "",
        "== Por juego ==", df["item_name"].value_counts().to_string(),
        "",
        "== Por rating (1=Recomendado, 0=No recomendado) ==", df["rating"].value_counts().to_string(),
        "",
        "== Por idioma ==", df["language"].value_counts().to_string(),
        "",
        "== Por año ==", df["date"].str[:4].value_counts().sort_index().to_string(),
        "",
        "== Por género ==",
        pd.Series(Counter(g for c in df["category"] for g in c.split("|", 1)[1].split(",") if g))
          .sort_values(ascending=False).to_string(),
        "",
        "== Por integrante ==", df["collected_by"].value_counts().to_string(),
        "",
        "== Longitud (palabras) ==", df["review"].str.split().str.len().describe().round(1).to_string(),
    ]
    OUT_STATS.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nEscrito: {OUT_TSV}\nEstadísticas: {OUT_STATS}")


if __name__ == "__main__":
    main()
