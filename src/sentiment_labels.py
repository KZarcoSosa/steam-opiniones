"""Etapa II, paso 1: etiquetas de sentimiento y particiones train/dev/test.

Uso:
    python main.py labels            # crea data/stage2/ (si no existe)
    python main.py labels --force    # vuelve a crearlo (¡cambia las particiones si cambió el corpus!)

Qué hace:
  1. Lee data/opinions_raw_corpus.tsv SIN modificarlo (corpus y benchmark de la Etapa I quedan congelados).
  2. Define la etiqueta y a partir de `rating`, que en este corpus es la recomendación explícita de
     Steam (voted_up): 1 = Recomendado -> "positive", 0 = No recomendado -> "negative".
  3. Hace las particiones 70/15/15 ANTES de ajustar cualquier vectorizador o modelo, con
     train_test_split(shuffle=True, random_state=0) estratificado por etiqueta.
  4. Escribe data/stage2/labeled_documents.tsv, split_report.txt y label_audit_sample.tsv
     (hoja para revisar a mano etiquetas posiblemente ambiguas).

"""
import argparse
import hashlib
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "opinions_raw_corpus.tsv"
OUT_DIR = ROOT / "data" / "stage2"
LABELED = OUT_DIR / "labeled_documents.tsv"
REPORT = OUT_DIR / "split_report.txt"
AUDIT = OUT_DIR / "label_audit_sample.tsv"

SEED = 0
DEV_SIZE = 0.15
TEST_SIZE = 0.15
# Mapeo documentado: la fuente (Steam) entrega una recomendación binaria, no una escala 1-5.
LABEL_MAP = {"1": "positive", "0": "negative"}
LABELS = ["negative", "positive"]
META_COLUMNS = ["item_id", "item_name", "date", "category", "source", "url", "source_id",
                "language", "playtime_hours", "votes_up"]

# Pistas simples para ENCONTRAR candidatas a revisión manual (no cambian ninguna etiqueta).
_RE_NEG_CUES = re.compile(
    r"\b(no lo recomiendo|no recomiendo|no lo compres|no compres|no lo compren|no compren|"
    r"decepcion\w*|basura|estafa|no vale|malisimo|pésimo|pesimo|horrible|arrepent\w*)\b", re.I)
# "no lo recomiendo" contiene "lo recomiendo": se elimina antes de buscar señales positivas.
_RE_NOT_RECOMMEND = re.compile(r"\bno (lo |se lo |te lo |les? )?(recomiendo|recomendaria|recomendaría|recomendable|recomendado)\b", re.I)
_RE_POS_CUES = re.compile(
    r"\b(lo recomiendo|muy recomendable|recomendado|obra maestra|excelente|me encanta|"
    r"lo mejor|adictivo|10/10|imprescindible|de lo mejor)\b", re.I)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw() -> pd.DataFrame:
    if not RAW.is_file():
        sys.exit(f"No existe {RAW}")
    df = pd.read_csv(RAW, sep="\t", dtype=str, keep_default_na=False)
    needed = {"review_id", "review", "rating"}
    if not needed.issubset(df.columns):
        sys.exit(f"Faltan columnas en el corpus crudo: {needed - set(df.columns)}")
    if df.review_id.duplicated().any():
        sys.exit("review_id duplicado en el corpus crudo")
    return df


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el mapeo y valida que no haya ratings faltantes o fuera de la escala esperada."""
    missing = df.rating.str.strip().eq("")
    unknown = ~df.rating.isin(LABEL_MAP) & ~missing
    if unknown.any():
        sys.exit(f"Ratings fuera del mapeo {set(LABEL_MAP)}: {df.rating[unknown].unique()}")
    if missing.any():
        # Etapa I garantiza 0 faltantes; si aparecieran, se excluyen y se reportan (no se inventan etiquetas).
        print(f"AVISO: {missing.sum()} reseñas sin rating; se excluyen de la Etapa II.")
        df = df[~missing].copy()
    df = df.copy()
    df["label"] = df.rating.map(LABEL_MAP)
    return df


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    """70/15/15 estratificado por etiqueta, con shuffle=True y random_state=0 (dos pasos)."""
    y = df["label"]
    idx_train, idx_tmp = train_test_split(
        df.index, test_size=DEV_SIZE + TEST_SIZE, shuffle=True, random_state=SEED, stratify=y)
    idx_dev, idx_test = train_test_split(
        idx_tmp, test_size=TEST_SIZE / (DEV_SIZE + TEST_SIZE), shuffle=True, random_state=SEED,
        stratify=y.loc[idx_tmp])
    df = df.copy()
    df["split"] = ""
    df.loc[idx_train, "split"] = "train"
    df.loc[idx_dev, "split"] = "dev"
    df.loc[idx_test, "split"] = "test"
    assert df.split.ne("").all()
    # Particiones disjuntas y completas
    assert len(set(idx_train) & set(idx_dev)) == len(set(idx_train) & set(idx_test)) == len(set(idx_dev) & set(idx_test)) == 0
    return df


def audit_sample(df: pd.DataFrame, n_random=20) -> pd.DataFrame:
    """Candidatas a etiqueta ambigua: (a) señales textuales que contradicen la etiqueta y
    (b) una muestra aleatoria (seed 0) para estimar qué tan bien refleja el texto la recomendación."""
    rows = []
    for _, r in df.iterrows():
        text = r["review"]
        if r["label"] == "positive" and _RE_NEG_CUES.search(text):
            rows.append((r, "etiqueta positive pero el texto contiene señales negativas"))
        elif r["label"] == "negative" and _RE_POS_CUES.search(_RE_NOT_RECOMMEND.sub(" ", text)):
            rows.append((r, "etiqueta negative pero el texto contiene señales positivas"))
    flagged = {r["review_id"] for r, _ in rows}
    rest = df[~df.review_id.isin(flagged)].sample(n=min(n_random, len(df) - len(flagged)), random_state=SEED)
    rows += [(r, "muestra aleatoria de control") for _, r in rest.iterrows()]
    out = pd.DataFrame({
        "document_id": [r["review_id"] for r, _ in rows],
        "item_name": [r["item_name"] for r, _ in rows],
        "split": [r["split"] for r, _ in rows],
        "label_from_rating": [r["label"] for r, _ in rows],
        "reason": [why for _, why in rows],
        "review": [re.sub(r"\s+", " ", r["review"]) for r, _ in rows],
        "manual_label": "",         # el equipo escribe: positive / negative / mixed / unclear
        "comment": "",
    })
    return out


def write_report(df: pd.DataFrame, audit: pd.DataFrame) -> str:
    lines = [
        "REPORTE DE ETIQUETAS Y PARTICIONES - Etapa II",
        f"Generado (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Corpus crudo: {RAW.name}  sha256={sha256(RAW)}",
        f"Python {platform.python_version()} | pandas {pd.__version__} | scikit-learn {sklearn.__version__}",
        f"Semilla: random_state={SEED}, shuffle=True, estratificado por etiqueta; "
        f"proporciones {1 - DEV_SIZE - TEST_SIZE:.0%}/{DEV_SIZE:.0%}/{TEST_SIZE:.0%}",
        "",
        "== Definición de la etiqueta ==",
        "Fuente: Steam (voted_up). rating=1 -> positive (Recomendado); rating=0 -> negative (No recomendado).",
        "No hay clase neutral porque la fuente solo ofrece recomendación binaria.",
        f"Reseñas sin rating: 0 | usadas: {len(df)}",
        "",
        "== Distribución de clases por partición ==",
        pd.crosstab(df.split, df.label, margins=True).reindex(["train", "dev", "test", "All"]).to_string(),
        "",
        "== Proporción de clases por partición (%) ==",
        (pd.crosstab(df.split, df.label, normalize="index") * 100).round(1).reindex(["train", "dev", "test"]).to_string(),
        "",
        "== Distribución por juego y partición (no se estratificó por juego; se reporta) ==",
        pd.crosstab(df.item_name, df.split).reindex(columns=["train", "dev", "test"]).to_string(),
        "",
        "== Revisión manual de etiquetas ==",
        f"label_audit_sample.tsv: {len(audit)} filas ({(audit.reason != 'muestra aleatoria de control').sum()} "
        f"candidatas por señales contradictorias + {(audit.reason == 'muestra aleatoria de control').sum()} de control).",
        "Pendiente: el equipo llena manual_label/comment. Las etiquetas NO se corrigen: se conserva la",
        "recomendación explícita de la fuente y se reporta el porcentaje de desacuerdo.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="sobrescribe data/stage2/ si ya existe")
    args = ap.parse_args()
    if LABELED.exists() and not args.force:
        sys.exit(f"{LABELED} ya existe: las particiones están congeladas. Usa --force solo si es imprescindible.")

    df = make_splits(make_labels(load_raw()))
    # Chequeo del PDF: cada clase con ejemplos suficientes en cada partición
    counts = pd.crosstab(df.split, df.label)
    if (counts.values < 20).any():
        print("AVISO: alguna clase tiene menos de 20 ejemplos en una partición:\n", counts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "document_id": df.review_id,
        "review": df.review,
        "text": df.review,          # X de los modelos (el PDF define text = review)
        "rating": df.rating,        # solo metadato: NUNCA entra a X
        "label": df.label,
        "split": df.split,
    })
    for col in META_COLUMNS:
        if col in df.columns:
            out[col] = df[col]
    out.to_csv(LABELED, sep="\t", index=False, encoding="utf-8", lineterminator="\n")

    audit = audit_sample(df)
    audit.to_csv(AUDIT, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    text = write_report(df, audit)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Escrito: {LABELED}\n         {AUDIT}\n         {REPORT}")


if __name__ == "__main__":
    main()