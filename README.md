# Práctica II — Recuperación de opiniones (PLN)

Corpus de reseñas de **Steam** en español, representadas con VSM (frecuencia, binaria, TF‑IDF; unigramas y bigramas) y recuperadas por similitud coseno.

## Estructura

```
config/apps.json          # qué juegos baja cada integrante
src/collect_steam.py      # 1. recolección (cada quien la corre con su nombre)
src/merge_corpus.py       # 2. une todo -> data/opinions_raw_corpus.tsv
data/raw/<integrante>/    # JSONL crudos (texto original) — esto es lo que suben
data/opinions_raw_corpus.tsv
data/corpus_stats.txt     # distribución por juego, rating, fecha, idioma
logs/                     # errores de adquisición (locales)
```

## Flujo de trabajo del equipo

```bash
git clone <repo> && cd <repo>
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Cada integrante, SOLO con su nombre:
python src/collect_steam.py --member kevin

git checkout -b datos-kevin
git add data/raw/kevin
git commit -m "Reseñas de kevin"
git push -u origin datos-kevin   # y abrir Pull Request
```

Como cada quien escribe solo en `data/raw/<su_nombre>/`, no hay conflictos de merge.

Cuando los tres hayan subido sus datos, **una sola persona** corre:

```bash
python src/merge_corpus.py --max-per-game 42
git add data/opinions_raw_corpus.tsv data/corpus_stats.txt
git commit -m "Corpus congelado v1" && git tag corpus-v1 && git push --tags
```

⚠️ Después de `corpus-v1` no se re-genera el TSV: los `review_id` son los `document_id` de `qrels.tsv`.

## Fuente y procedimiento (para el reporte)

- API pública de reseñas de Steam: `store.steampowered.com/appreviews/<appid>?json=1` (sin login, sin API key, sin evadir CAPTCHAs).
- Nombre y géneros: `store.steampowered.com/api/appdetails`.
- Idiomas: `latam` (español de Latinoamérica) y `spanish` (español de España).
- Pausa de 1.5 s entre peticiones, reintentos con espera ante errores/429, log de errores en `logs/`.
- Se descartan reseñas de menos de 15 palabras.
- `rating`: Steam solo da recomendado (1) / no recomendado (0).
- No se guarda el identificador del autor.
