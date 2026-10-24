"""
Muestra las 5 mejores y las 5 peores consultas a partir de la salida del
evaluador oficial del curso (no modifica ese script ni su salida).

Uso:
    python main.py casos                 # 5 mejores y 5 peores de las 15
    python main.py casos --nivel 1       # solo fáciles (1), medias (2) o difíciles (3),
                                         # de mejor a peor
    python main.py casos --por-nivel     # los tres niveles, cada uno de mejor a peor
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "results" / "evaluacion_profesor" / "evaluation_results.tsv"
QUERIES = ROOT / "data" / "benchmark" / "queries.tsv"
ANALYSIS = ROOT / "docs" / "analisis_errores.tsv"
LEVEL = {"1": "fácil", "2": "media", "3": "difícil"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nivel", choices=["1", "2", "3"], help="1 fácil, 2 media, 3 difícil")
    ap.add_argument("--por-nivel", action="store_true", help="ranking dentro de cada nivel")
    ap.add_argument("--archivo", default=str(EVAL), help="evaluation_results.tsv del evaluador")
    args = ap.parse_args()
    path = Path(args.archivo)
    if not path.is_file():
        sys.exit(f"No existe {path}. Corre antes el evaluador del profesor (ver README).")
    res = pd.read_csv(path, sep="\t", dtype={"query_id": str})
    mean = res[res.query_id == "MEAN"].iloc[0]
    res = res[res.query_id != "MEAN"]
    q = pd.read_csv(QUERIES, sep="\t", dtype=str).set_index("query_id")
    causes = {}
    if ANALYSIS.is_file():
        a = pd.read_csv(ANALYSIS, sep="\t", dtype=str, keep_default_na=False)
        causes = dict(zip(a.query_id, a.causa_principal))
    res = res.assign(consulta=res.query_id.map(q["query"]),
                     nivel=res.query_id.map(q["difficulty_level"]).map(LEVEL))

    def show(title, df):
        print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
        print(f"{'ID':4} {'Nivel':8} {'P@10':>5} {'RR':>5} {'nDCG':>5}  Consulta")
        for r in df.itertuples():
            print(f"{r.query_id:4} {r.nivel:8} {r._2:5.2f} {r.RR:5.2f} {r._6:5.2f}  {r.consulta}")
            if r.query_id in causes:
                print(f"{'':31}-> {causes[r.query_id]}")

    order = ["nDCG@10", "P@10"]
    if args.nivel or args.por_nivel:
        for lvl in ([args.nivel] if args.nivel else ["1", "2", "3"]):
            sub = res[res.nivel == LEVEL[lvl]].sort_values(order, ascending=False)
            show(f"NIVEL {lvl} ({LEVEL[lvl].upper()}) - de mejor a peor por nDCG@10", sub)
            m = sub[["P@10", "R@10", "RR", "nDCG@10"]].mean()
            print(f"{'':4} Promedio del nivel: P@10 {m['P@10']:.3f} | R@10 {m['R@10']:.3f} | "
                  f"MRR {m['RR']:.3f} | nDCG@10 {m['nDCG@10']:.3f}")
            print(f"{'':4} Mejor: {sub.iloc[0].query_id}   Peor: {sub.iloc[-1].query_id}")
        return
    show("5 MEJORES CONSULTAS (por nDCG@10)", res.sort_values(order, ascending=False).head(5))
    show("5 PEORES CONSULTAS (por nDCG@10)", res.sort_values(order).head(5))

    print(f"\n{'=' * 100}\nPROMEDIO POR NIVEL DE DIFICULTAD\n{'=' * 100}")
    lv = res.groupby("nivel")[["P@10", "R@10", "RR", "nDCG@10"]].mean().reindex(["fácil", "media", "difícil"])
    print(lv.rename(columns={"RR": "MRR"}).to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nSISTEMA COMPLETO: P@10 {mean['P@10']:.4f} | R@10 {mean['R@10']:.4f} | "
          f"F1@10 {mean['F1@10']:.4f} | MRR {mean['RR']:.4f} | nDCG@10 {mean['nDCG@10']:.4f}")


if __name__ == "__main__":
    main()
