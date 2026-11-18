"""Etapa II, paso 4: evaluación FINAL en test (una sola vez).

Uso:
    python main.py final-test                      # usa la configuración que seleccionó el ejecutor
    python main.py final-test --model MultinomialNB   # el equipo elige la mejor configuración de un modelo

Qué hace (secciones 6 y 8 del PDF de la Etapa II):
  1. Lee la configuración seleccionada en dev (data/stage2/best_by_model.json -> "overall").
  2. Reajusta SOLO ese modelo y esa configuración (hiperparámetros congelados) con train + dev.
  3. Evalúa en test por primera y única vez; añade las filas con experiment_id="test" a
     classification_metrics.tsv y classification_predictions.tsv.
  4. Guarda el pipeline final en models/sentiment/ para la interfaz de búsqueda.
  5. Crea data/stage2/final_test_done.json: a partir de ahí, experiments y este comando se niegan a
     correr otra vez (para que nadie "afine" contra test). --force lo permite, pero invalida la evaluación.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from sentiment_experiments import (BEST, FINAL_DONE, LABELS, METRIC_COLUMNS, METRICS, OUT, PREDICTIONS,  # noqa: E402
                                   ROOT, build_pipeline, describe, load_all, metrics_row, write_tsv)

MODEL_DIR = ROOT / "models" / "sentiment"
PIPELINE_FILE = MODEL_DIR / "final_pipeline.joblib"
CONFIG_FILE = MODEL_DIR / "final_config.json"
TEST_REPORT = OUT / "test_report.txt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["MultinomialNB", "LogisticRegression", "LinearSVC"],
                    help="elegir la mejor configuración (en dev) de este modelo en lugar de la seleccionada automáticamente")
    ap.add_argument("--yes", action="store_true", help="no pedir confirmación")
    ap.add_argument("--force", action="store_true", help="repetir aunque ya exista una evaluación final (invalida el protocolo)")
    args = ap.parse_args()

    if FINAL_DONE.exists() and not args.force:
        sys.exit("Ya hay una evaluación final en test (final_test_done.json). El protocolo permite UNA sola.")
    if not BEST.is_file():
        sys.exit("Falta best_by_model.json. Primero: python main.py experiments")

    data = json.loads(BEST.read_text(encoding="utf-8"))
    if args.model:
        best = {**data["by_model"][args.model], "model": args.model}
        selection = f"elegida por el equipo: mejor configuración en dev de {args.model}"
    else:
        best = data["overall"]
        selection = "automática del ejecutor (macro-F1 en dev; desempate: más simple, luego CV en train)"
    spec = best["spec"]
    norm, enh = describe(spec)
    print(f"Selección: {selection}")
    print("Configuración seleccionada en dev (congelada):")
    print(f"  {best['experiment_id']} | {best['model']} | {norm} | {enh} | {spec['rep']} ngram={tuple(spec['ngram'])} "
          f"| {spec['clf_params']} | dev macro-F1={best['dev_macro_f1']:.4f}")
    if not args.yes:
        if input("\nEsto evalúa en TEST y solo se hace una vez. Escribe SI para continuar: ").strip() != "SI":
            sys.exit("Cancelado.")

    df = load_all()
    trdv, te = df[df.split.isin(["train", "dev"])], df[df.split == "test"]
    assert set(trdv.split) == {"train", "dev"} and set(te.split) == {"test"}
    assert not set(trdv.document_id) & set(te.document_id)

    pipe = build_pipeline(spec)
    pipe.fit(trdv.text.tolist(), trdv.label.tolist())          # train + dev, sin test
    y_pred = pipe.predict(te.text.tolist())
    m = metrics_row("test", best["model"], "test", te.label.tolist(), list(y_pred))

    # Añadir a los archivos de entrega (quitando filas "test" previas si se usó --force)
    mt = pd.read_csv(METRICS, sep="\t", dtype=str, keep_default_na=False)
    mt = mt[mt.split != "test"]
    write_tsv(pd.concat([mt, pd.DataFrame([m])[METRIC_COLUMNS].astype(str)], ignore_index=True), METRICS, float_cols=False)
    pr = pd.read_csv(PREDICTIONS, sep="\t", dtype=str, keep_default_na=False)
    pr = pr[pr.split != "test"]
    new = pd.DataFrame({"experiment_id": "test", "model": best["model"], "document_id": te.document_id.tolist(),
                        "gold_label": te.label.tolist(), "predicted_label": list(y_pred), "split": "test"})
    write_tsv(pd.concat([pr, new], ignore_index=True), PREDICTIONS, float_cols=False)
    # los números del archivo quedan redondeados a 4 decimales
    mt2 = pd.read_csv(METRICS, sep="\t", dtype=str, keep_default_na=False)
    for c in ["accuracy", "macro_precision", "macro_recall", "macro_f1"] + [c for c in mt2.columns if c.startswith(("precision_", "recall_", "f1_"))]:
        mt2[c] = mt2[c].map(lambda v: f"{float(v):.4f}")
    write_tsv(mt2, METRICS, float_cols=False)

    report = (f"EVALUACIÓN FINAL EN TEST (única)\nGenerado (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
              f"Modelo: {best['model']} | selección en dev: {best['experiment_id']} (macro-F1 dev={best['dev_macro_f1']:.4f})\n"
              f"Criterio de selección: {selection}\n"
              f"Normalización: {norm} | Mejoras: {enh} | {spec['rep']} ngram={tuple(spec['ngram'])} | {spec['clf_params']}\n"
              f"Reajustado con train+dev: {len(trdv)} documentos | test: {len(te)} documentos\n\n"
              f"{classification_report(te.label.tolist(), y_pred, labels=LABELS, digits=4, zero_division=0)}\n"
              f"Matriz de confusión (filas=real, columnas=predicho; orden {LABELS}):\n"
              f"{confusion_matrix(te.label.tolist(), y_pred, labels=LABELS)}\n")
    TEST_REPORT.write_text(report, encoding="utf-8")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, PIPELINE_FILE)
    CONFIG_FILE.write_text(json.dumps({"experiment_id": best["experiment_id"], "model": best["model"], "spec": spec,
                                       "trained_on": "train+dev", "labels": LABELS}, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    FINAL_DONE.write_text(json.dumps({
        "done_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "selected_experiment": best["experiment_id"],
        "model": best["model"], "selection": selection, "test_macro_f1": round(m["macro_f1"], 4),
        "best_by_model_sha256": hashlib.sha256(BEST.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    print("\n" + report)
    print(f"Pipeline guardado: {PIPELINE_FILE}\nEscrito: {TEST_REPORT}\n         {METRICS}\n         {PREDICTIONS}")
    print("\nA partir de ahora NO cambies preprocesamiento, modelo ni hiperparámetros: la evaluación en test ya se usó.")


if __name__ == "__main__":
    main()