"""Etapa II, paso 5: interfaz de sentimiento (sección 10 del PDF de la Etapa II).

Uso:
    python main.py sentiment predict                 # sentimiento predicho para todo el corpus (data/stage2/predicted_sentiment.tsv)
    python main.py sentiment predict --text "..."    # clasifica una reseña nueva con el mismo pipeline
    python main.py sentiment metrics                 # métricas por clase, matrices de confusión (dev por modelo y test)
    python main.py sentiment experiments [--stage B] [--model LinearSVC] [--top 15]
    python main.py sentiment errors [--split dev|test] [-n 10]

La búsqueda (python main.py search ...) usa predicted_sentiment.tsv para mostrar el sentimiento predicho
y para filtrar con --sentimiento positivo|negativo.

Nota: el pipeline final se entrenó con train+dev, así que las predicciones de esos documentos son "en muestra";
solo las de test (columna in_sample=no) son predicciones sobre reseñas no vistas.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
OUT = ROOT / "data" / "stage2"
LABELED = OUT / "labeled_documents.tsv"
PREDICTED = OUT / "predicted_sentiment.tsv"
METRICS = OUT / "classification_metrics.tsv"
PREDICTIONS = OUT / "classification_predictions.tsv"
EXPERIMENTS = OUT / "experiments.tsv"
DEV_ERRORS = OUT / "dev_errors.tsv"
PIPELINE = ROOT / "models" / "sentiment" / "final_pipeline.joblib"
CONFIG = ROOT / "models" / "sentiment" / "final_config.json"
ES = {"positive": "positivo", "negative": "negativo"}


def _need(path, hint):
    if not Path(path).is_file():
        sys.exit(f"Falta {path}\n{hint}")


def _read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _short(text, n=350):
    return " ".join(str(text).split())[:n]


# ----------------------------------------------------------------------------- usado por retrieve.py
def load_predictions():
    """{document_id: (etiqueta, margen, split)} o None si todavía no se generó predicted_sentiment.tsv."""
    if not PREDICTED.is_file():
        return None
    d = _read(PREDICTED)
    return {r.document_id: (r.predicted_label, float(r.margin), r.split) for r in d.itertuples()}


# ----------------------------------------------------------------------------- predict
def _load_pipeline():
    _need(PIPELINE, "Primero ejecuta: python main.py final-test")
    sys.path.insert(0, str(SRC))                    # el pipeline guardado referencia sentiment_features
    import joblib
    return joblib.load(PIPELINE)


def _margin(pipe, texts):
    """Distancia con signo a la frontera (positivo => clase 'positive'). Para NB usa log-odds."""
    if hasattr(pipe, "decision_function"):
        return np.asarray(pipe.decision_function(texts), dtype=float)
    p = pipe.predict_proba(texts)[:, list(pipe.classes_).index("positive")]
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def cmd_predict(args):
    pipe = _load_pipeline()
    if args.text:
        label = pipe.predict([args.text])[0]
        m = _margin(pipe, [args.text])[0]
        print(f"Sentimiento predicho: {ES[label]}  (margen={m:+.3f}; >0 positivo, <0 negativo)")
        return
    _need(LABELED, "Primero ejecuta: python main.py labels")
    d = _read(LABELED)
    pred = pipe.predict(d.text.tolist())
    margin = _margin(pipe, d.text.tolist())
    out = pd.DataFrame({"document_id": d.document_id, "predicted_label": pred, "margin": np.round(margin, 4),
                        "gold_label": d.label, "split": d.split,
                        "in_sample": np.where(d.split == "test", "no", "si")})
    out.to_csv(PREDICTED, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    print(f"Escrito {PREDICTED} ({len(out)} documentos)")
    print(out.groupby("predicted_label").size().rename(index=ES).to_string())
    t = out[out.split == "test"]
    print(f"\nSolo test (no visto por el modelo): exactitud={(t.predicted_label == t.gold_label).mean():.4f} sobre {len(t)} documentos")
    print("Ahora la búsqueda muestra el sentimiento: python main.py search --query \"...\" [--sentimiento positivo|negativo]")


# ----------------------------------------------------------------------------- metrics
def cmd_metrics(args):
    _need(METRICS, "Primero ejecuta: python main.py experiments")
    m = _read(METRICS)
    print("Métricas (orden de clases: negative, positive). Dev: mejor configuración de cada modelo; test: evaluación final única.\n")
    for _, r in m.iterrows():
        print(f"[{r.split}] {r.experiment_id:>5} {r.model:19s} exactitud={r.accuracy} | macro P={r.macro_precision} "
              f"R={r.macro_recall} F1={r.macro_f1}")
        print(f"        negative: P={r.precision_negative} R={r.recall_negative} F1={r.f1_negative} (n={r.support_negative}) | "
              f"positive: P={r.precision_positive} R={r.recall_positive} F1={r.f1_positive} (n={r.support_positive})")
        print(f"        matriz de confusión (filas=real, columnas=predicho): [[{r.cm_tn} {r.cm_fp}] [{r.cm_fn} {r.cm_tp}]]\n")


# ----------------------------------------------------------------------------- experiments
def cmd_experiments(args):
    _need(EXPERIMENTS, "Primero ejecuta: python main.py experiments")
    e = pd.read_csv(EXPERIMENTS, sep="\t")
    if args.stage:
        e = e[e.stage == args.stage.upper()]
    if args.model:
        e = e[e.method == args.model]
    e = e.sort_values("dev_macro_f1", ascending=False).head(args.top)
    cols = ["experiment_id", "stage", "method", "normalization", "enhancement", "representation", "hyperparameters",
            "dev_macro_f1", "cv_macro_f1_train", "cv_difference_vs_baseline", "best_for_model", "selected"]
    cols = [c for c in cols if c in e.columns]
    pd.set_option("display.width", 250, "display.max_colwidth", 38)
    print(e[cols].round(4).fillna("").to_string(index=False))
    print(f"\n({len(e)} filas, ordenadas por macro-F1 en dev. Log completo: {EXPERIMENTS})")


# ----------------------------------------------------------------------------- errors
def cmd_errors(args):
    n = args.n
    if args.split == "dev":
        _need(DEV_ERRORS, "Primero ejecuta: python main.py experiments")
        d = _read(DEV_ERRORS)
        d["n_models_wrong"] = d.n_models_wrong.astype(int)
        d = d.sort_values(["n_models_wrong", "document_id"], ascending=[False, True]).head(n)
        pred_cols = [c for c in d.columns if c.startswith("pred_")]
        print(f"Errores en dev (los que fallan más modelos primero). Mostrando {len(d)}.\n")
        for _, r in d.iterrows():
            preds = " ".join(f"{c[5:]}={ES[r[c]]}" for c in pred_cols)
            print(f"doc={r.document_id} | {r.item_name} | real={ES[r.gold_label]} | {preds} | fallan {r.n_models_wrong} de {len(pred_cols)}")
            print(f"    {_short(r.review)}\n")
        return
    _need(PREDICTIONS, "Primero ejecuta: python main.py final-test")
    p = _read(PREDICTIONS)
    p = p[(p.experiment_id == "test") & (p.gold_label != p.predicted_label)]
    if p.empty:
        sys.exit("No hay errores de test registrados (¿ya ejecutaste final-test?).")
    lab = _read(LABELED).set_index("document_id")
    marg = load_predictions() or {}
    p = p.assign(abs_margin=[abs(marg[d][1]) if d in marg else 0.0 for d in p.document_id])
    p = p.sort_values("abs_margin", ascending=False).head(n)
    print(f"Errores en test ({p.model.iloc[0]}): primero los de mayor margen (errores 'seguros'). Mostrando {len(p)}.")
    print("Solo para análisis: NO se usan para ajustar nada (el protocolo ya cerró test).\n")
    for _, r in p.iterrows():
        row = lab.loc[r.document_id]
        print(f"doc={r.document_id} | {row.item_name} | real={ES[r.gold_label]} | predicho={ES[r.predicted_label]} | margen={marg.get(r.document_id, (0, 0))[1]:+.3f}")
        print(f"    {_short(row.text)}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--text")
    p.set_defaults(fn=cmd_predict)
    sub.add_parser("metrics").set_defaults(fn=cmd_metrics)
    p = sub.add_parser("experiments")
    p.add_argument("--stage", choices=["A", "B", "C", "D", "a", "b", "c", "d"])
    p.add_argument("--model", choices=["MultinomialNB", "LogisticRegression", "LinearSVC"])
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(fn=cmd_experiments)
    p = sub.add_parser("errors")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(fn=cmd_errors)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()