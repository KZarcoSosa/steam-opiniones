"""Aplica UNA vez los cambios mínimos de la Etapa II a archivos existentes:
  * src/retrieve.py : opción --sentimiento y sentimiento predicho en los resultados de `search`
  * main.py         : comandos `sentiment` (-> src/sentiment_app.py) e `incremental` (-> src/sentiment_incremental.py)
Hace copia de seguridad (.bak) y se niega a modificar si el texto esperado no coincide.

Uso (desde la raíz del proyecto):   python tools/aplicar_parche_etapa2.py
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RETRIEVE = ROOT / "src" / "retrieve.py"
MAIN = ROOT / "main.py"
MARK = "# --- Etapa II"

R_EDITS = [
    # 1) helper de sentimiento (antes de search)
    ("def search(query, content='review', features='unigram', weight='tfidf', top=10):\n",
     MARK + ": sentimiento predicho (opcional) ---\n"
     "SENT_ES = {'positive': 'positivo', 'negative': 'negativo'}\n\n\n"
     "def _sentiment_map():\n"
     "    \"\"\"{review_id: (etiqueta, margen, split)} o None si aún no se ejecutó: python main.py sentiment predict\"\"\"\n"
     "    try:\n"
     "        import sentiment_app\n"
     "    except ImportError:\n"
     "        return None\n"
     "    return sentiment_app.load_predictions()\n\n\n"
     "def search(query, content='review', features='unigram', weight='tfidf', top=10, sentiment=None):\n"),
    # 2) filtrar el ranking completo por sentimiento predicho
    ("    ranking = rank_normalized_query(qnorm, df, vectorizer, matrix)\n"
     "    raw = pd.read_csv(RAW, sep='\\t', dtype=str, keep_default_na=False).set_index('review_id')\n",
     "    ranking = rank_normalized_query(qnorm, df, vectorizer, matrix)\n"
     "    sent = _sentiment_map()\n"
     "    if sentiment:  # Etapa II: se filtra el ranking completo y luego se toman los `top`\n"
     "        if sent is None:\n"
     "            raise RuntimeError('No hay predicciones de sentimiento. Ejecuta: python main.py sentiment predict')\n"
     "        wanted = {'positivo': 'positive', 'negativo': 'negative'}[sentiment]\n"
     "        ranking = [(d, s) for d, s in ranking if sent[d][0] == wanted]\n"
     "    raw = pd.read_csv(RAW, sep='\\t', dtype=str, keep_default_na=False).set_index('review_id')\n"),
    # 3) campos en cada resultado
    ("                        'review': row.review, 'url': row.url})\n",
     "                        'review': row.review, 'url': row.url,\n"
     "                        'sentiment': SENT_ES[sent[doc_id][0]] if sent else None})\n"),
    # 4) impresión
    ("    print(f'Comparación: {args.content} | {args.features} | {args.weight}')\n    for item in results:\n",
     "    print(f'Comparación: {args.content} | {args.features} | {args.weight}')\n"
     "    if getattr(args, 'sentimiento', None):\n"
     "        print(f'Filtro Etapa II: solo opiniones con sentimiento predicho {args.sentimiento}')\n"
     "    for item in results:\n"),
    ("              f\"| {item['item_name']} | rating={item['rating']}{mark}\\n    {short}\")\n",
     "              f\"| {item['item_name']} | rating={item['rating']}\"\n"
     "              f\"{' | sentimiento predicho=' + item['sentiment'] if item.get('sentiment') else ''}{mark}\\n    {short}\")\n"),
    # 5) argumento y llamada en main()
    ("    parser.add_argument('--top', type=int, default=10)\n    args = parser.parse_args()\n",
     "    parser.add_argument('--top', type=int, default=10)\n"
     "    parser.add_argument('--sentimiento', choices=['positivo', 'negativo'],\n"
     "                        help='Etapa II: mostrar solo opiniones con ese sentimiento predicho')\n"
     "    args = parser.parse_args()\n"),
    ("    qnorm, results = search(args.query, args.content, args.features, args.weight, args.top)\n",
     "    qnorm, results = search(args.query, args.content, args.features, args.weight, args.top, args.sentimiento)\n"),
]


def patch_retrieve():
    text = RETRIEVE.read_text(encoding="utf-8")
    if MARK in text:
        print("retrieve.py: ya estaba parchado (no se cambia).")
        return True
    for old, _ in R_EDITS:
        if text.count(old) != 1:
            print(f"retrieve.py: no encuentro (o aparece más de una vez) este texto esperado:\n{old}\n"
                  "No se modificó nada. Avísame para ajustar el parche a tu versión.")
            return False
    for old, new in R_EDITS:
        text = text.replace(old, new)
    shutil.copy(RETRIEVE, RETRIEVE.with_suffix(".py.bak"))
    RETRIEVE.write_text(text, encoding="utf-8")
    print("retrieve.py: parchado (copia en retrieve.py.bak).")
    return True


def add_command(text, name, script, anchors):
    """Añade "name": "script" a COMMANDS justo después de la primera línea ancla que exista."""
    if re.search(rf"[\"']{re.escape(name)}[\"']\s*:", text):
        return text, "ya existe"
    for anchor in anchors:
        m = re.search(rf"(?m)^(\s*)([\"']){re.escape(anchor)}\2\s*:.*$", text)
        if m:
            line = m.group(0).rstrip()
            if not line.endswith(","):
                line += ","
            text = text[:m.start()] + line + f'\n{m.group(1)}"{name}": "{script}",' + text[m.end():]
            return text, "añadido"
    return text, "sin ancla"


def patch_main():
    text = MAIN.read_text(encoding="utf-8")
    original, ok = text, True
    for name, script, anchors in [("sentiment", "sentiment_app.py", ["final-test", "experiments"]),
                                  ("incremental", "sentiment_incremental.py", ["sentiment", "final-test"])]:
        text, status = add_command(text, name, script, anchors)
        if status == "sin ancla":
            print(f'main.py: no encontré dónde insertar "{name}". Agrega a mano en COMMANDS:\n    "{name}": "{script}",')
            ok = False
        else:
            print(f"main.py: comando '{name}' {status}.")
    if text != original:
        shutil.copy(MAIN, MAIN.with_suffix(".py.bak"))
        MAIN.write_text(text, encoding="utf-8")
    return ok


if __name__ == "__main__":
    ok = patch_retrieve() & patch_main()
    sys.exit(0 if ok else 1)