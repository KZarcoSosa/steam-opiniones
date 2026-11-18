"""Etapa II: impacto de cada preprocesamiento, de forma INCREMENTAL (comentario del profesor).

Uso:
    python main.py incremental                  # cadena completa S0..S6 con los 3 clasificadores
    python main.py incremental --core           # solo S0-S3: base, repeticiones, negación, léxicos
    python main.py incremental --models LinearSVC MultinomialNB
    python main.py incremental --show 5         # imprime N reseñas transformadas paso a paso
    python main.py incremental --quick          # prueba rápida (CV 3x2)
    python main.py incremental --no-variants    # no escribe los corpus variantes

Qué hace:
  1. CADENA INCREMENTAL. Parte de una base (tokenización + minúsculas) y va SUMANDO un factor a la vez:
        S0 base -> S1 +repeticiones -> S2 +negación -> S3 +léxicos -> S4 +lemas -> S5 +ortografía -> S6 +palabras funcionales
     Para cada paso mide el macro-F1 y la diferencia contra el paso anterior y contra la base.
  2. ABLACIÓN "DEJAR UNO FUERA". Parte de la cadena completa y quita un factor a la vez: cuánto se pierde sin él.
  3. CORPUS VARIANTES. Guarda en data/stage2/variants/ el corpus tal como queda en cada paso (+ INDEX.tsv con
     cuántos documentos cambia cada paso), para poder inspeccionarlos.

Control experimental: en TODOS los pasos se mantienen fijos la representación y los hiperparámetros de la línea
base de cada clasificador (tomados de experiments.tsv); lo único que cambia es el preprocesamiento. Las
comparaciones son PAREADAS: las mismas particiones de validación cruzada (5x5, solo train) para todos los pasos, y se
reporta la diferencia por partición con la prueba t corregida de Nadeau y Bengio (apropiada para CV repetida).
Dev se reporta como medida descriptiva adicional. TEST NO SE USA.
Es un análisis descriptivo del impacto: no se usa para cambiar el modelo ya evaluado en test.
"""
import argparse
import ast
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from sentiment_experiments import (CLASSIFIERS, EXPERIMENTS, OUT, SEED, _fallback_lemmas, build_pipeline,  # noqa: E402
                                   item_words, load_all, load_dev_data)
from sentiment_features import LexiconFeatures, TextPreprocessor, spacy_available  # noqa: E402

VARIANTS_DIR = OUT / "variants"
INCREMENTAL = OUT / "incremental_ablation.tsv"
LOO = OUT / "leave_one_out.tsv"
SUMMARY = OUT / "incremental_summary.txt"
LEX_BLOCKS = ["sel", "liwc", "emoji", "negation"]


# ------------------------------------------------------------------------------------ pasos
def step_definitions(items):
    """(id, nombre corto, descripción, cambios_preproc, bloques_léxicos, necesita_spacy, necesita_corrector)"""
    return [
        ("S1_repeticiones", "repeticiones", "normalizar letras alargadas y palabras repetidas",
         {"repeats": True}, [], False, False),
        ("S2_negacion", "negación", "marcar el alcance de la negación (NEG_)",
         {"negation": True}, [], False, False),
        ("S3_lexicos", "léxicos", "agregar características de léxicos (SEL, LIWC, emojis, negadores)",
         {}, LEX_BLOCKS, False, False),
        ("S4_lemas", "lemas", "lematizar con spaCy",
         {"lemmatize": True}, [], True, False),
        ("S5_ortografia", "ortografía", "corrección ortográfica (pyspellchecker)",
         {"spelling": True, "protected_words": items}, [], False, True),
        ("S6_funcionales", "palabras funcionales", "quitar artículos, preposiciones, conjunciones y pronombres (negadores protegidos)",
         {"remove_function_words": True, "protect_negators": True}, [], True, False),
    ]


def build_chain(items, core_only, have_spacy, have_speller):
    """Cadena acumulativa: cada paso conserva todo lo anterior y suma un factor."""
    chain = [{"id": "S0_base", "factor": "base", "desc": "tokenización + minúsculas (línea base)",
              "preproc": {}, "lex": []}]
    skipped = []
    for sid, factor, desc, upd, blocks, need_spacy, need_speller in step_definitions(items):
        if core_only and sid not in ("S1_repeticiones", "S2_negacion", "S3_lexicos"):
            continue
        if need_spacy and not have_spacy:
            skipped.append((sid, "spaCy/es_core_news_sm no disponible"))
            continue
        if need_speller and not have_speller:
            skipped.append((sid, "falta pyspellchecker"))
            continue
        prev = chain[-1]
        pre = deepcopy(prev["preproc"])
        pre.update(upd)
        lex = list(prev["lex"]) + [b for b in blocks if b not in prev["lex"]]
        chain.append({"id": sid, "factor": factor, "desc": desc, "preproc": pre, "lex": lex})
    return chain, skipped


def without_factor(step_full, chain, factor):
    """Configuración completa pero SIN el factor indicado (dejar uno fuera)."""
    removed = next(s for s in chain if s["factor"] == factor)
    pre = deepcopy(step_full["preproc"])
    lex = list(step_full["lex"])
    # quita exactamente lo que ese paso había añadido
    prev = chain[chain.index(removed) - 1]
    for k in list(removed["preproc"]):
        if removed["preproc"].get(k) != prev["preproc"].get(k):
            pre.pop(k, None)
    lex = [b for b in lex if b not in (set(removed["lex"]) - set(prev["lex"]))]
    return pre, lex


# ------------------------------------------------------------------------------------ línea base por modelo
def baseline_settings(models):
    """Representación e hiperparámetros de la línea base de cada clasificador (experiments.tsv)."""
    defaults = {"MultinomialNB": ("binary", (1, 1), {"alpha": 1.0}),
                "LogisticRegression": ("binary", (1, 2), {"C": 1.0}),
                "LinearSVC": ("binary", (1, 2), {"C": 0.1})}
    out = {}
    e = pd.read_csv(EXPERIMENTS, sep="\t") if EXPERIMENTS.is_file() else None
    for m in models:
        got = None
        if e is not None:
            b = e[(e.stage == "B") & (e.method == m)]
            if len(b):
                row = e[e.experiment_id == b.baseline_id.iloc[0]]
                if len(row):
                    r = row.iloc[0]
                    try:
                        rep = r.representation.split()[0]
                        ng = tuple(ast.literal_eval(r.representation.split("ngram=")[1]))
                        params = {}
                        for kv in str(r.hyperparameters).split(";"):
                            if "=" in kv:
                                k, v = kv.split("=", 1)
                                try:
                                    params[k.strip()] = ast.literal_eval(v.strip())
                                except Exception:
                                    params[k.strip()] = v.strip()
                        got = (rep, ng, params, r.experiment_id)
                    except Exception:
                        got = None
        if got is None:
            rep, ng, params = defaults[m]
            print(f"AVISO: no pude leer la línea base de {m} en experiments.tsv; uso {rep} {ng} {params}")
            got = (rep, ng, params, "defecto")
        out[m] = got
    return out


def make_spec(model, settings, preproc, lex):
    rep, ng, params, _ = settings[model]
    return {"preproc": deepcopy(preproc), "rep": rep, "ngram": list(ng), "lex_blocks": list(lex),
            "clf": model, "clf_params": dict(params)}


# ------------------------------------------------------------------------------------ estadística
def paired_stats(a, b, test_over_train):
    """Diferencia por partición d = b - a. Prueba t corregida (Nadeau y Bengio) para CV repetida."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    k = len(d)
    mean = float(d.mean())
    var = float(d.var(ddof=1)) if k > 1 else 0.0
    if var <= 1e-18:
        return mean, mean, mean, (1.0 if abs(mean) < 1e-12 else 0.0), int((d > 0).sum()), k
    se = np.sqrt((1.0 / k + test_over_train) * var)
    t = mean / se
    p = float(2 * (1 - stats.t.cdf(abs(t), k - 1)))
    h = stats.t.ppf(0.975, k - 1) * se
    return mean, mean - float(h), mean + float(h), p, int((d > 0).sum()), k


def verdict(mean, p):
    if abs(mean) < 0.005:
        return "sin cambio apreciable"
    if p < 0.05:
        return "MEJORA (p<0.05)" if mean > 0 else "EMPEORA (p<0.05)"
    return "tendencia a mejorar (no concluyente)" if mean > 0 else "tendencia a empeorar (no concluyente)"


def n_features(pipe):
    try:
        if "vec" in pipe.named_steps:
            return len(pipe.named_steps["vec"].vocabulary_)
        return len(pipe.named_steps["feats"].transformer_list[0][1].named_steps["vec"].vocabulary_)
    except Exception:
        return np.nan


# ------------------------------------------------------------------------------------ evaluación
def evaluate(spec, X_tr, y_tr, X_dv, y_dv, cv):
    pipe = build_pipeline(spec)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=1)
    pipe = build_pipeline(spec)
    pipe.fit(X_tr, y_tr)
    dev = f1_score(y_dv, pipe.predict(X_dv), average="macro")
    return scores, float(dev), n_features(pipe)


def run_chain(models, chain, settings, data, cv, ratio):
    X_tr, y_tr, X_dv, y_dv = data
    rows, kept = [], {}
    for model in models:
        print(f"\n=== {model} | línea base {settings[model][3]}: {settings[model][0]} ngram={settings[model][1]} {settings[model][2]} ===")
        prev = base = None
        for st in chain:
            spec = make_spec(model, settings, st["preproc"], st["lex"])
            scores, dev, nfeat = evaluate(spec, X_tr, y_tr, X_dv, y_dv, cv)
            kept[(model, st["id"])] = scores
            row = {"classifier": model, "step_id": st["id"], "added_factor": st["factor"], "description": st["desc"],
                   "representation": f"{settings[model][0]} ngram={settings[model][1]}",
                   "hyperparameters": "; ".join(f"{k}={v}" for k, v in settings[model][2].items()),
                   "n_features": nfeat, "cv_macro_f1_mean": scores.mean(), "cv_macro_f1_std": scores.std(ddof=1),
                   "dev_macro_f1": dev}
            if prev is None:
                base = (scores, dev)
                row.update({"delta_prev_cv": np.nan, "delta_prev_ci_low": np.nan, "delta_prev_ci_high": np.nan,
                            "delta_prev_p": np.nan, "folds_improved_vs_prev": "", "verdict_vs_prev": "(base)",
                            "delta_base_cv": np.nan, "delta_base_p": np.nan, "dev_delta_prev": np.nan, "dev_delta_base": np.nan})
            else:
                m, lo, hi, p, w, k = paired_stats(prev[0], scores, ratio)
                mb, _, _, pb, _, _ = paired_stats(base[0], scores, ratio)
                row.update({"delta_prev_cv": m, "delta_prev_ci_low": lo, "delta_prev_ci_high": hi, "delta_prev_p": p,
                            "folds_improved_vs_prev": f"{w}/{k}", "verdict_vs_prev": verdict(m, p),
                            "delta_base_cv": mb, "delta_base_p": pb,
                            "dev_delta_prev": dev - prev[1], "dev_delta_base": dev - base[1]})
            rows.append(row)
            prev = (scores, dev)
            dp = "" if np.isnan(row["delta_prev_cv"]) else f" | Δprev {row['delta_prev_cv']:+.4f} (p={row['delta_prev_p']:.2f}, {row['folds_improved_vs_prev']}) {row['verdict_vs_prev']}"
            print(f"  {st['id']:<17s} CV {scores.mean():.4f}±{scores.std(ddof=1):.4f} | dev {dev:.4f} | {nfeat} rasgos{dp}")
    return pd.DataFrame(rows), kept


def run_loo(models, chain, settings, data, cv, ratio, kept):
    X_tr, y_tr, X_dv, y_dv = data
    full = chain[-1]
    rows = []
    for model in models:
        full_scores = kept[(model, full["id"])]
        spec_full = make_spec(model, settings, full["preproc"], full["lex"])
        pipe = build_pipeline(spec_full)
        pipe.fit(X_tr, y_tr)
        dev_full = float(f1_score(y_dv, pipe.predict(X_dv), average="macro"))
        print(f"\n=== {model}: dejar uno fuera (completo = {full['id']}: CV {full_scores.mean():.4f}, dev {dev_full:.4f}) ===")
        for st in chain[1:]:
            pre, lex = without_factor(full, chain, st["factor"])
            spec = make_spec(model, settings, pre, lex)
            scores, dev, _ = evaluate(spec, X_tr, y_tr, X_dv, y_dv, cv)
            contrib, lo, hi, p, w, k = paired_stats(scores, full_scores, ratio)    # completo - sin el factor
            rows.append({"classifier": model, "removed_factor": st["factor"], "step_id": st["id"],
                         "cv_macro_f1_without": scores.mean(), "cv_macro_f1_full": full_scores.mean(),
                         "contribution_cv": contrib, "ci_low": lo, "ci_high": hi, "p": p,
                         "folds_where_factor_helps": f"{w}/{k}", "verdict": verdict(contrib, p),
                         "dev_macro_f1_without": dev, "dev_macro_f1_full": dev_full, "contribution_dev": dev_full - dev})
            print(f"  sin {st['factor']:<22s} CV {scores.mean():.4f} | aporte del factor {contrib:+.4f} (p={p:.2f}) {verdict(contrib, p)}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ corpus variantes
def write_variants(chain, df_all, show=0):
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    texts = df_all.text.tolist()
    index, prev_text, outputs = [], None, {}
    for st in chain:
        pre = TextPreprocessor(**st["preproc"])
        out = pre.transform(texts)
        outputs[st["id"]] = out
        n_changed = int(sum(a != b for a, b in zip(out, prev_text))) if prev_text is not None else 0
        fname = f"corpus_{st['id']}.tsv"
        v = pd.DataFrame({"document_id": df_all.document_id, "split": df_all.split, "label": df_all.label,
                          "text_original": df_all.text.map(lambda t: " ".join(str(t).split())), "text_variant": out})
        if st["lex"]:                                  # el paso de léxicos añade columnas numéricas
            lf = LexiconFeatures(blocks=tuple(st["lex"]), lemma_fn=None if spacy_available() else _fallback_lemmas)
            F = lf.transform(texts)
            for j, name in enumerate(lf.get_feature_names_out()):
                v[name] = np.round(F[:, j], 4)
        v.to_csv(VARIANTS_DIR / fname, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
        toks = [len(t.split()) for t in out]
        train_vocab = set(w for t, s in zip(out, df_all.split) if s == "train" for w in t.split())
        index.append({"step_id": st["id"], "file": fname, "description": st["desc"],
                      "docs_changed_vs_prev": n_changed, "pct_changed_vs_prev": round(100 * n_changed / len(out), 1),
                      "mean_tokens": round(float(np.mean(toks)), 1), "vocab_train": len(train_vocab),
                      "extra_feature_columns": ",".join(st["lex"]) if st["lex"] else ""})
        prev_text = out
    pd.DataFrame(index).to_csv(VARIANTS_DIR / "INDEX.tsv", sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    # ejemplos paso a paso: documentos donde algún paso cambia el texto
    ids = list(range(len(texts)))
    changed = [i for i in ids if len({outputs[s["id"]][i] for s in chain}) > 1]
    pick = changed[: max(show, 8)]
    lines = ["EJEMPLOS: la misma reseña en cada paso de la cadena incremental (solo reseñas que cambian en algún paso)\n"]
    for i in pick:
        lines.append(f"--- document_id={df_all.document_id.iloc[i]} | {df_all.item_name.iloc[i]} | etiqueta={df_all.label.iloc[i]}")
        lines.append(f"  ORIGINAL        : {' '.join(str(texts[i]).split())[:260]}")
        last = None
        for st in chain:
            t = outputs[st['id']][i]
            mark = "" if last is None or t != last else "  (sin cambio)"
            lines.append(f"  {st['id']:<16s}: {t[:260]}{mark}")
            last = t
        lines.append("")
    (VARIANTS_DIR / "examples.txt").write_text("\n".join(lines), encoding="utf-8")
    if show:
        print("\n" + "\n".join(lines[: 1 + 8 * min(show, len(pick))]))
    return pd.DataFrame(index)


# ------------------------------------------------------------------------------------ gráficas y resumen
def make_plots(inc, loo):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib no instalado: no se generan las gráficas; instala con: pip install matplotlib)")
        return
    models = list(inc.classifier.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        d = inc[inc.classifier == m]
        x = np.arange(len(d))
        ax.errorbar(x, d.cv_macro_f1_mean, yerr=d.cv_macro_f1_std, marker="o", capsize=3, label="CV train (media ± desv.)")
        ax.plot(x, d.dev_macro_f1, marker="s", linestyle="--", label="dev")
        ax.set_xticks(x)
        ax.set_xticklabels([s.split("_", 1)[1] for s in d.step_id], rotation=35, ha="right")
        ax.set_title(m)
        ax.grid(alpha=.3)
    axes[0][0].set_ylabel("macro-F1")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Cadena incremental: macro-F1 al sumar cada preprocesamiento")
    fig.tight_layout()
    fig.savefig(OUT / "incremental_ablation.png", dpi=150)
    plt.close(fig)
    if loo is not None and len(loo):
        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), sharey=True, squeeze=False)
        for ax, m in zip(axes[0], models):
            d = loo[loo.classifier == m]
            y = np.arange(len(d))
            ax.barh(y, d.contribution_cv, xerr=[d.contribution_cv - d.ci_low, d.ci_high - d.contribution_cv], capsize=3)
            ax.axvline(0, color="k", lw=.8)
            ax.set_yticks(y)
            ax.set_yticklabels(d.removed_factor)
            ax.set_title(m)
            ax.grid(alpha=.3)
        axes[0][0].set_xlabel("aporte del factor al macro-F1 (CV, IC 95 %)")
        fig.suptitle("Dejar uno fuera: cuánto se pierde sin cada factor")
        fig.tight_layout()
        fig.savefig(OUT / "leave_one_out.png", dpi=150)
        plt.close(fig)
    print(f"Gráficas: {OUT / 'incremental_ablation.png'} | {OUT / 'leave_one_out.png'}")


def write_summary(inc, loo, vindex, skipped, cvdesc):
    L = ["IMPACTO DE CADA PREPROCESAMIENTO (cadena incremental)", cvdesc,
         "Representación e hiperparámetros fijos por clasificador; solo cambia el preprocesamiento. Test no se usa.", ""]
    if skipped:
        L.append("Pasos omitidos: " + "; ".join(f"{s} ({why})" for s, why in skipped) + "\n")
    for m, d in inc.groupby("classifier", sort=False):
        L.append(f"== {m} ==")
        for _, r in d.iterrows():
            dp = "" if pd.isna(r.delta_prev_cv) else (f"Δprev {r.delta_prev_cv:+.4f} [{r.delta_prev_ci_low:+.4f},{r.delta_prev_ci_high:+.4f}] "
                                                      f"p={r.delta_prev_p:.2f} ({r.folds_improved_vs_prev}) -> {r.verdict_vs_prev}; Δbase {r.delta_base_cv:+.4f}")
            L.append(f"  {r.step_id:<17s} CV {r.cv_macro_f1_mean:.4f}  dev {r.dev_macro_f1:.4f}  {dp}")
        L.append("")
    if loo is not None and len(loo):
        L.append("DEJAR UNO FUERA (aporte = completo - sin el factor)")
        for m, d in loo.groupby("classifier", sort=False):
            L.append(f"== {m} ==")
            for _, r in d.iterrows():
                L.append(f"  sin {r.removed_factor:<22s} aporte CV {r.contribution_cv:+.4f} [{r.ci_low:+.4f},{r.ci_high:+.4f}] p={r.p:.2f} -> {r.verdict}")
            L.append("")
    if vindex is not None:
        L.append("EFECTO SOBRE EL CORPUS (cuántas reseñas cambia cada paso)")
        L.append(vindex[["step_id", "docs_changed_vs_prev", "pct_changed_vs_prev", "mean_tokens", "vocab_train"]].to_string(index=False))
    SUMMARY.write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", action="store_true", help="solo S0-S3 (base, repeticiones, negación, léxicos)")
    ap.add_argument("--models", nargs="+", choices=list(CLASSIFIERS), default=list(CLASSIFIERS))
    ap.add_argument("--quick", action="store_true", help="CV 3x2 en lugar de 5x5")
    ap.add_argument("--show", type=int, default=0, help="imprimir N reseñas paso a paso")
    ap.add_argument("--no-variants", action="store_true", help="no escribir data/stage2/variants/")
    ap.add_argument("--no-loo", action="store_true", help="omitir 'dejar uno fuera'")
    args = ap.parse_args()

    X_tr, y_tr, X_dv, y_dv, df_tr, df_dv = load_dev_data()          # test NO se carga
    items = item_words(pd.concat([df_tr, df_dv]))
    have_spacy = spacy_available()
    try:
        import spellchecker  # noqa: F401
        have_speller = True
    except ImportError:
        have_speller = False
    chain, skipped = build_chain(items, args.core, have_spacy, have_speller)
    for sid, why in skipped:
        print(f"(paso omitido {sid}: {why})")
    settings = baseline_settings(args.models)
    splits, reps = (3, 2) if args.quick else (5, 5)
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=reps, random_state=SEED)
    ratio = 1.0 / (splits - 1)                                       # n_test / n_train en cada partición
    cvdesc = f"CV: RepeatedStratifiedKFold({splits} pliegues x {reps} repeticiones, seed={SEED}) sobre train ({len(X_tr)} reseñas), {splits * reps} particiones pareadas"
    print(cvdesc)
    print("Cadena:", " -> ".join(s["id"] for s in chain))

    data = (X_tr, y_tr, X_dv, y_dv)
    inc, kept = run_chain(args.models, chain, settings, data, cv, ratio)
    loo = None
    if not args.no_loo and len(chain) > 2:
        loo = run_loo(args.models, chain, settings, data, cv, ratio, kept)
    OUT.mkdir(parents=True, exist_ok=True)
    inc.round(4).to_csv(INCREMENTAL, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    if loo is not None:
        loo.round(4).to_csv(LOO, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    vindex = None
    if not args.no_variants or args.show:
        vindex = write_variants(chain, load_all(), args.show)
        print("\nEfecto de cada paso sobre el corpus:\n" + vindex[["step_id", "docs_changed_vs_prev", "pct_changed_vs_prev", "mean_tokens", "vocab_train"]].to_string(index=False))
    write_summary(inc, loo, vindex, skipped, cvdesc)
    make_plots(inc, loo)
    print(f"\nEscrito en {OUT}: incremental_ablation.tsv" + (", leave_one_out.tsv" if loo is not None else "")
          + ", incremental_summary.txt" + ("" if args.no_variants else ", variants/"))
    print("Recuerda: es un análisis descriptivo del impacto; el modelo final ya se evaluó en test y no se cambia.")


if __name__ == "__main__":
    main()