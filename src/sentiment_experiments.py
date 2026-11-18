"""Etapa II, paso 3: experimentos supervisados (SOLO train y dev; test no se carga nunca aquí).

Uso:
    python main.py experiments            # corre toda la cuadrícula (unos minutos)
    python main.py experiments --quick    # versión reducida para probar que todo funciona
    python main.py experiments --force    # si ya hay un resultado final en test, permite sobrescribir (¡no recomendado!)

Diseño (secciones 5-9 del PDF de la Etapa II):
  Etapa A  Representación x clasificador, con preprocesamiento mínimo:
           {binaria, frecuencia, TF-IDF} x {(1,1), (1,2)} x {MultinomialNB, LogisticRegression, LinearSVC}.
           De aquí sale la LÍNEA BASE de cada clasificador (su mejor representación en dev).
  Etapa B  Un factor a la vez sobre esa línea base: 3 normalizaciones y 9 mejoras lingüísticas
           (negación, repeticiones, ortografía, léxicos SEL / LIWC / emojis, combinaciones).
  Etapa C  Desbalance (class_weight / fit_prior=False) sobre la mejor configuración de cada modelo.
  Etapa D  Diagnóstico: clasificador que usa SOLO las características de léxicos (no es candidato).
  Todos los hiperparámetros se ajustan con GridSearchCV (5 pliegues estratificados x 2 repeticiones, macro-F1)
  usando SOLO train; después cada candidato se evalúa en dev. La selección es por macro-F1 en dev; si dos
  configuraciones empatan, gana la MÁS SIMPLE (menos mejoras activas) y luego la de mayor CV en train.
  experiments.tsv guarda también la diferencia contra la línea base medida en CV (más estable que dev).

Salidas en data/stage2/: experiments.tsv, classification_metrics.tsv, classification_predictions.tsv,
best_by_model.json, environment.json, dev_reports.txt, dev_errors.tsv.
"""
import argparse
import hashlib
import json
import platform
import re
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from sentiment_features import (NEGATORS, WORD_RE, LexiconFeatures, TextPreprocessor,  # noqa: E402
                                spacy_available, strip_accents)

ROOT = SRC.parent
OUT = ROOT / "data" / "stage2"
LABELED = OUT / "labeled_documents.tsv"
EXPERIMENTS = OUT / "experiments.tsv"
METRICS = OUT / "classification_metrics.tsv"
PREDICTIONS = OUT / "classification_predictions.tsv"
BEST = OUT / "best_by_model.json"
ENV = OUT / "environment.json"
DEV_REPORTS = OUT / "dev_reports.txt"
DEV_ERRORS = OUT / "dev_errors.tsv"
FINAL_DONE = OUT / "final_test_done.json"

SEED = 0
LABELS = ["negative", "positive"]
REPRESENTATIONS = ["binary", "frequency", "tfidf"]
NGRAMS = [(1, 1), (1, 2)]

METRIC_COLUMNS = ["experiment_id", "model", "split", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                  "precision_negative", "recall_negative", "f1_negative", "support_negative",
                  "precision_positive", "recall_positive", "f1_positive", "support_positive",
                  "cm_tn", "cm_fp", "cm_fn", "cm_tp"]


# ------------------------------------------------------------------------------------ datos
def _fallback_lemmas(text):
    """Solo se usa si spaCy no está instalado (p. ej. pruebas): SEL sin lematizar."""
    return [strip_accents(w.lower()) for w in WORD_RE.findall(text)]


def load_all():
    if not LABELED.is_file():
        sys.exit(f"No existe {LABELED}. Primero ejecuta: python main.py labels")
    return pd.read_csv(LABELED, sep="\t", dtype=str, keep_default_na=False)


def load_dev_data():
    """Devuelve (X_train, y_train, X_dev, y_dev, df_train, df_dev). El conjunto test NO se carga."""
    df = load_all()
    df = df[df.split.isin(["train", "dev"])].copy()
    assert "test" not in set(df.split)
    tr, dv = df[df.split == "train"], df[df.split == "dev"]
    return tr.text.tolist(), tr.label.tolist(), dv.text.tolist(), dv.label.tolist(), tr, dv


def item_words(df):
    """Palabras de los nombres de juegos: el corrector ortográfico no debe tocarlas (solo metadato, no la etiqueta)."""
    words = set()
    for name in df.item_name.unique():
        words |= {w.lower() for w in WORD_RE.findall(name) if len(w) >= 3}
    return sorted(words)


# ------------------------------------------------------------------------------------ pipelines
CLASSIFIERS = {
    "MultinomialNB": {"grid": {"clf__alpha": [0.1, 0.5, 1.0]}, "quick": {"clf__alpha": [0.5, 1.0]}},
    "LogisticRegression": {"grid": {"clf__C": [0.1, 1.0, 10.0]}, "quick": {"clf__C": [1.0, 10.0]}},
    "LinearSVC": {"grid": {"clf__C": [0.1, 1.0, 10.0]}, "quick": {"clf__C": [0.1, 1.0]}},
}


def make_classifier(name, params):
    p = dict(params)
    if name == "MultinomialNB":
        return MultinomialNB(**p)
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=2000, random_state=SEED, **p)
    if name == "LinearSVC":
        return LinearSVC(random_state=SEED, max_iter=10000, **p)
    raise ValueError(name)


def make_vectorizer(rep, ngram):
    kw = dict(token_pattern=r"\S+", lowercase=False, ngram_range=tuple(ngram))   # los tokens ya vienen separados por espacios
    if rep == "tfidf":
        return TfidfVectorizer(**kw)
    return CountVectorizer(binary=(rep == "binary"), **kw)


def build_pipeline(spec):
    """spec (serializable en JSON): preproc, rep, ngram, lex_blocks, clf, clf_params."""
    pre = TextPreprocessor(**spec["preproc"])
    text_branch = Pipeline([("pre", pre), ("vec", make_vectorizer(spec["rep"], spec["ngram"]))])
    clf = make_classifier(spec["clf"], spec.get("clf_params", {}))
    if spec.get("lexicon_only"):                              # diagnóstico: solo características de léxicos
        lemma_fn = None if spacy_available() else _fallback_lemmas
        return Pipeline([("lex", LexiconFeatures(blocks=tuple(spec["lex_blocks"]), lemma_fn=lemma_fn)),
                         ("scale", MaxAbsScaler()), ("clf", clf)])
    if spec.get("lex_blocks"):
        lemma_fn = None if spacy_available() else _fallback_lemmas
        lex = Pipeline([("lex", LexiconFeatures(blocks=tuple(spec["lex_blocks"]), lemma_fn=lemma_fn)),
                        ("scale", MaxAbsScaler())])         # ajustado solo con train, dentro del Pipeline
        feats = FeatureUnion([("text", text_branch), ("lex", lex)])
        return Pipeline([("feats", feats), ("clf", clf)])
    return Pipeline([("pre", text_branch.steps[0][1]), ("vec", text_branch.steps[1][1]), ("clf", clf)])


def describe(spec):
    flags = spec["preproc"]
    parts = []
    if flags.get("lemmatize"):
        parts.append("lematización")
    if flags.get("remove_function_words"):
        parts.append("sin palabras funcionales" + (" (negadores protegidos)" if flags.get("protect_negators", True)
                                                      else " (como Etapa I)"))
    norm = "+".join(parts) if parts else "tokenización+minúsculas"
    if spec.get("lexicon_only"):
        norm = "(sin texto: solo características de léxicos)"
    enh = []
    if flags.get("negation"):
        enh.append("negación")
    if flags.get("repeats"):
        enh.append("repeticiones")
    if flags.get("spelling"):
        enh.append("ortografía")
    if spec.get("lex_blocks"):
        enh.append("léxicos[" + ",".join(spec["lex_blocks"]) + "]")
    return norm, "+".join(enh) if enh else "ninguna"


def complexity(spec):
    """Cuántas mejoras/ajustes extra tiene la configuración (para desempatar a favor de la más simple)."""
    f = spec["preproc"]
    c = sum(bool(f.get(k)) for k in ("lemmatize", "remove_function_words", "negation", "repeats", "spelling"))
    c += int(bool(spec.get("lex_blocks")))
    cp = spec.get("clf_params", {})
    c += int("class_weight" in cp or "fit_prior" in cp)
    return c


def selection_key(rec):
    """Mayor macro-F1 en dev (a 4 decimales); empate -> menos complejidad; empate -> mayor CV en train."""
    return (round(rec["dev_macro_f1"], 4), -complexity(json.loads(rec["spec_json"])), rec["cv_macro_f1_train"])


# ------------------------------------------------------------------------------------ métricas
def metrics_row(experiment_id, model, split, y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    row = {"experiment_id": experiment_id, "model": model, "split": split,
           "accuracy": accuracy_score(y_true, y_pred),
           "macro_precision": p.mean(), "macro_recall": r.mean(), "macro_f1": f.mean(),
           "cm_tn": int(cm[0, 0]), "cm_fp": int(cm[0, 1]), "cm_fn": int(cm[1, 0]), "cm_tp": int(cm[1, 1])}
    for i, lab in enumerate(LABELS):
        row.update({f"precision_{lab}": p[i], f"recall_{lab}": r[i], f"f1_{lab}": f[i], f"support_{lab}": int(s[i])})
    return row


def write_tsv(df, path, float_cols=True):
    df = df.copy()
    if float_cols:
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].round(4)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_environment(args):
    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    env = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "spacy": ver("spacy"), "spacy_model": "es_core_news_sm " + str(ver("es_core_news_sm")),
        "pyspellchecker": ver("pyspellchecker"), "openpyxl": ver("openpyxl"),
        "seed": SEED, "cv": ("StratifiedKFold(n_splits=3, shuffle=True, random_state=0)" if args.quick
               else "RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)"),
        "scoring_for_tuning": "f1_macro", "selection_metric": "macro-F1 en dev (desempate: configuración más simple; luego CV en train)",
        "labeled_documents_sha256": sha256(LABELED),
        "spacy_disponible": spacy_available(),
        "nota": "Si spacy_disponible=false, las variantes con spaCy se omitieron y SEL se calculó sin lematizar.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    ENV.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------------------------ ejecutor
class Runner:
    def __init__(self, quick):
        self.quick = quick
        self.X_tr, self.y_tr, self.X_dv, self.y_dv, self.df_tr, self.df_dv = load_dev_data()
        self.items = item_words(pd.concat([self.df_tr, self.df_dv]))
        self.records, self.dev_preds, self.counter = [], {}, 0
        self.cv = (StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED) if quick
                   else RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED))
        self.have_spacy = spacy_available()
        try:
            import spellchecker  # noqa: F401
            self.have_speller = True
        except ImportError:
            self.have_speller = False

    def run(self, stage, factor, clf_name, spec, baseline_id=None, note=""):
        """Ajusta hiperparámetros con CV en train, evalúa en dev, registra todo."""
        self.counter += 1
        eid = f"C{self.counter:03d}"
        t0 = time.time()
        pipe = build_pipeline(spec)
        grid = CLASSIFIERS[clf_name]["quick" if self.quick else "grid"]
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=self.cv, n_jobs=1, refit=True)
        gs.fit(self.X_tr, self.y_tr)
        y_pred = gs.predict(self.X_dv)
        m = metrics_row(eid, clf_name, "dev", self.y_dv, y_pred)
        i = gs.best_index_
        spec_done = deepcopy(spec)
        spec_done["clf_params"] = {**spec.get("clf_params", {}),
                                   **{k.replace("clf__", ""): v for k, v in gs.best_params_.items()}}
        norm, enh = describe(spec)
        imb = "class_weight=balanced" if spec_done["clf_params"].get("class_weight") == "balanced" else \
              ("fit_prior=False" if spec_done["clf_params"].get("fit_prior") is False else "ninguno")
        base = next((r for r in self.records if r["experiment_id"] == baseline_id), None)
        rec = {
            "experiment_id": eid, "stage": stage, "factor_changed": factor, "normalization": norm,
            "representation": f"{spec['rep']} ngram={tuple(spec['ngram'])}", "enhancement": enh,
            "imbalance": imb, "method": clf_name,
            "hyperparameters": "; ".join(f"{k}={v}" for k, v in spec_done["clf_params"].items()),
            "cv_macro_f1_train": gs.cv_results_["mean_test_score"][i],
            "cv_std": gs.cv_results_["std_test_score"][i],
            "dev_accuracy": m["accuracy"], "dev_macro_precision": m["macro_precision"],
            "dev_macro_recall": m["macro_recall"], "dev_macro_f1": m["macro_f1"],
            "dev_metric": f"macro-F1={m['macro_f1']:.4f}",
            "baseline_id": baseline_id or "",
            "difference_vs_baseline": (m["macro_f1"] - base["dev_macro_f1"]) if base else np.nan,
            "cv_difference_vs_baseline": (gs.cv_results_["mean_test_score"][i] - base["cv_macro_f1_train"]) if base else np.nan,
            "best_for_model": "", "selected": "", "notes": note, "seconds": round(time.time() - t0, 1),
            "spec_json": json.dumps(spec_done, ensure_ascii=False, sort_keys=True),
        }
        self.records.append(rec)
        self.dev_preds[eid] = (list(y_pred), m)
        d = rec["difference_vs_baseline"]
        print(f"  {eid} [{stage}] {clf_name:19s} {factor:28s} dev macro-F1={m['macro_f1']:.4f}"
              + (f" ({d:+.4f} vs base)" if not np.isnan(d) else "") + f"  {gs.best_params_}")
        return rec

    # --- etapas -----------------------------------------------------------------------------
    def stage_a(self):
        print("\n== Etapa A: representación x clasificador (preprocesamiento mínimo) ==")
        for clf in CLASSIFIERS:
            for rep in REPRESENTATIONS:
                for ng in NGRAMS:
                    spec = {"preproc": {}, "rep": rep, "ngram": list(ng), "lex_blocks": [], "clf": clf, "clf_params": {}}
                    self.run("A", f"representación {rep} {ng}", clf, spec)
        self.baseline = {}
        for clf in CLASSIFIERS:
            cands = [r for r in self.records if r["stage"] == "A" and r["method"] == clf]
            best = max(cands, key=selection_key)
            self.baseline[clf] = best
            print(f"  -> línea base {clf}: {best['experiment_id']} {best['representation']} (dev {best['dev_macro_f1']:.4f})")

    def variants(self):
        """(nombre, factor, cambios_preproc, bloques_léxicos, necesita_spacy, necesita_corrector)"""
        return [
            ("norm_etapa1", "normalización", dict(lemmatize=True, remove_function_words=True, protect_negators=False), [], True, False),
            ("norm_lemas", "normalización", dict(lemmatize=True), [], True, False),
            ("norm_lemas_sin_funcionales_protegiendo_negadores", "normalización",
             dict(lemmatize=True, remove_function_words=True, protect_negators=True), [], True, False),
            ("mejora_negacion", "mejora", dict(negation=True), [], False, False),
            ("mejora_repeticiones", "mejora", dict(repeats=True), [], False, False),
            ("mejora_ortografia", "mejora", dict(spelling=True, protected_words=self.items), [], False, True),
            ("mejora_lexico_SEL", "mejora", {}, ["sel"], False, False),
            ("mejora_lexico_LIWC", "mejora", {}, ["liwc"], False, False),
            ("mejora_emojis", "mejora", {}, ["emoji"], False, False),
            ("mejora_lexicos_todos", "mejora", {}, ["sel", "liwc", "emoji", "negation"], False, False),
            ("negacion+lexicos_todos", "combinación", dict(negation=True), ["sel", "liwc", "emoji", "negation"], False, False),
            ("lemas+negacion", "combinación", dict(lemmatize=True, negation=True), [], True, False),
        ]

    def stage_b(self):
        print("\n== Etapa B: un factor a la vez sobre la línea base de cada clasificador ==")
        for clf, base in self.baseline.items():
            base_spec = json.loads(base["spec_json"])
            print(f"\n-- {clf} (base {base['experiment_id']}: {base['representation']}) --")
            singles = {}
            for name, factor, upd, blocks, need_spacy, need_speller in self.variants():
                if need_spacy and not self.have_spacy:
                    print(f"  (omitido {name}: spaCy no disponible)")
                    continue
                if need_speller and not self.have_speller:
                    print(f"  (omitido {name}: falta pyspellchecker)")
                    continue
                spec = deepcopy(base_spec)
                spec["preproc"].update(upd)
                spec["lex_blocks"] = blocks
                if need_speller:
                    print("  (ortografía: la primera vez puede tardar unos minutos; después usa caché)", flush=True)
                rec = self.run("B", name, clf, spec, baseline_id=base["experiment_id"])
                if name in ("mejora_negacion", "mejora_repeticiones", "mejora_ortografia", "mejora_lexico_SEL",
                            "mejora_lexico_LIWC", "mejora_emojis"):
                    singles[name] = (rec, upd, blocks)
            positives = [(n, v) for n, v in singles.items() if v[0]["difference_vs_baseline"] > 0]
            if len(positives) >= 2:
                spec = deepcopy(base_spec)
                blocks = []
                for _, (_, upd, b) in positives:
                    spec["preproc"].update(upd)
                    blocks += [x for x in b if x not in blocks]
                spec["lex_blocks"] = blocks
                self.run("B", "combinación_de_mejoras_positivas", clf, spec, baseline_id=base["experiment_id"],
                         note="junta las mejoras individuales con diferencia > 0 en dev: " + ", ".join(n for n, _ in positives))

    def best_so_far(self, clf):
        c = [r for r in self.records if r["method"] == clf and r["stage"] in ("A", "B", "C")]
        return max(c, key=selection_key)

    def stage_c(self):
        print("\n== Etapa C: desbalance (las clases están casi equilibradas; se compara de todos modos) ==")
        dist = pd.Series(self.y_tr).value_counts().to_dict()
        print(f"  distribución en train: {dist}")
        for clf in CLASSIFIERS:
            best = self.best_so_far(clf)
            spec = json.loads(best["spec_json"])
            if clf == "MultinomialNB":
                spec["clf_params"]["fit_prior"] = False          # análogo de class_weight para NB
            else:
                spec["clf_params"]["class_weight"] = "balanced"
            spec["clf_params"] = {k: v for k, v in spec["clf_params"].items() if k not in ("alpha", "C")}  # se vuelven a ajustar
            self.run("C", "desbalance", clf, spec, baseline_id=best["experiment_id"],
                     note=f"clases en train: {dist}; solo entrenamiento, dev no se toca")

    def stage_d(self):
        print("\n== Etapa D (diagnóstico): clasificador que usa SOLO los léxicos (no compite en la selección) ==")
        spec = {"preproc": {}, "rep": "binary", "ngram": [1, 1], "lex_blocks": ["sel", "liwc", "emoji", "negation"],
                "lexicon_only": True, "clf": "LogisticRegression", "clf_params": {}}
        self.run("D", "solo_léxicos", "LogisticRegression", spec,
                 note="diagnóstico: poder predictivo de los léxicos por sí solos; no es candidato")

    # --- selección y salidas -------------------------------------------------------------------
    def finish(self):
        for clf in CLASSIFIERS:
            self.best_so_far(clf)["best_for_model"] = "yes"
        bests = [r for r in self.records if r["best_for_model"] == "yes"]
        overall = max(bests, key=selection_key)
        overall["selected"] = "yes"
        exp = pd.DataFrame(self.records).drop(columns=["spec_json"])
        write_tsv(exp, EXPERIMENTS)

        rows, preds, reports = [], [], []
        dev_ids = self.df_dv.document_id.tolist()
        for r in bests:
            y_pred, m = self.dev_preds[r["experiment_id"]]
            rows.append(m)
            preds += [{"experiment_id": r["experiment_id"], "model": r["method"], "document_id": d,
                       "gold_label": g, "predicted_label": p, "split": "dev"}
                      for d, g, p in zip(dev_ids, self.y_dv, y_pred)]
            reports.append(f"=== {r['experiment_id']} {r['method']} | {r['normalization']} | {r['enhancement']} | "
                           f"{r['representation']} | {r['hyperparameters']}\n"
                           f"{classification_report(self.y_dv, y_pred, labels=LABELS, digits=4, zero_division=0)}\n"
                           f"Matriz de confusión (filas=real, columnas=predicho; orden {LABELS}):\n"
                           f"{confusion_matrix(self.y_dv, y_pred, labels=LABELS)}\n")
        write_tsv(pd.DataFrame(rows)[METRIC_COLUMNS], METRICS)
        write_tsv(pd.DataFrame(preds)[["experiment_id", "model", "document_id", "gold_label", "predicted_label", "split"]],
                  PREDICTIONS)
        DEV_REPORTS.write_text("\n".join(reports), encoding="utf-8")
        self.write_dev_errors(bests)
        BEST.write_text(json.dumps({
            "overall": {"experiment_id": overall["experiment_id"], "model": overall["method"],
                        "dev_macro_f1": overall["dev_macro_f1"], "spec": json.loads(overall["spec_json"])},
            "by_model": {r["method"]: {"experiment_id": r["experiment_id"], "dev_macro_f1": r["dev_macro_f1"],
                                       "spec": json.loads(r["spec_json"])} for r in bests},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n== Resumen: mejor configuración por modelo (macro-F1 en dev) ==")
        for r in sorted(bests, key=lambda r: -r["dev_macro_f1"]):
            star = "  <-- seleccionado" if r["selected"] == "yes" else ""
            print(f"  {r['experiment_id']} {r['method']:19s} dev macro-F1={r['dev_macro_f1']:.4f} "
                  f"(CV train {r['cv_macro_f1_train']:.4f}±{r['cv_std']:.4f}) | {r['normalization']} | {r['enhancement']} | "
                  f"{r['representation']}{star}")
        n_dev = len(self.y_dv)
        print(f"\nAVISO: dev tiene solo {n_dev} documentos; un documento mueve el macro-F1 ~{1 / n_dev:.3f}. "
              f"Diferencias menores a ~0.03-0.05 pueden ser ruido.")
        print(f"Escrito en {OUT}: experiments.tsv, classification_metrics.tsv, classification_predictions.tsv,"
              f" best_by_model.json, dev_reports.txt, dev_errors.tsv, environment.json")

    def write_dev_errors(self, bests):
        by_model = {}
        for r in bests:
            by_model[r["method"]] = self.dev_preds[r["experiment_id"]][0]
        d = self.df_dv.reset_index(drop=True).copy()
        for model, pred in by_model.items():
            d[f"pred_{model}"] = pred
        wrong = np.zeros(len(d), dtype=int)
        for model in by_model:
            wrong += (d[f"pred_{model}"] != d["label"]).to_numpy().astype(int)
        d["n_models_wrong"] = wrong
        d = d[d.n_models_wrong > 0].copy()
        words = lambda t: [strip_accents(w.lower()) for w in WORD_RE.findall(t)]
        d["n_negators"] = d.text.map(lambda t: sum(w in NEGATORS for w in words(t)))
        d["has_emoji_or_emoticon"] = d.text.map(lambda t: int(bool(re.search(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]|[:;=][\-o^']?[)(DdPp]", t))))
        d["n_caps_words"] = d.text.map(lambda t: len(re.findall(r"\b[A-ZÁÉÍÓÚÑ]{3,}\b", t)))
        d["n_words"] = d.text.map(lambda t: len(words(t)))
        d["review"] = d.text.map(lambda t: re.sub(r"\s+", " ", t))
        cols = ["document_id", "item_name", "label", *[f"pred_{m}" for m in by_model], "n_models_wrong",
                "n_negators", "has_emoji_or_emoticon", "n_caps_words", "n_words", "review"]
        write_tsv(d.sort_values("n_models_wrong", ascending=False)[cols].rename(columns={"label": "gold_label"}), DEV_ERRORS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="cuadrícula y validación cruzada reducidas (prueba rápida)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if FINAL_DONE.exists() and not args.force:
        sys.exit("Ya existe la evaluación final en test (final_test_done.json). Volver a correr experimentos "
                 "sobrescribiría los resultados en dev que la respaldan. Usa --force solo si sabes lo que haces.")
    if not args.quick and not spacy_available():
        print("AVISO: spaCy/es_core_news_sm no disponible: se omiten las variantes con lematización.")
    t0 = time.time()
    write_environment(args)
    runner = Runner(args.quick)
    runner.stage_a()
    runner.stage_b()
    runner.stage_c()
    runner.stage_d()
    runner.finish()
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()