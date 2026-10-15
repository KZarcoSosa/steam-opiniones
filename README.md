# Práctica II — Recuperación de opiniones (PLN)

Corpus de reseñas de **Steam** en español, representadas con el Modelo de Espacio Vectorial (frecuencia, binaria, TF‑IDF; unigramas y bigramas) y recuperadas por similitud coseno.

Equipo: Kevin, Angel, Erick. Guía detallada de cómo funciona todo: [`docs/GUIA.md`](docs/GUIA.md).

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Comandos (sección 10 de la práctica)

| Paso | Comando | Genera |
|---|---|---|
| 1. Recolección | `python main.py collect --member kevin` | `data/raw/kevin/*.jsonl` |
| Unir corpus | `python main.py merge` | `data/opinions_raw_corpus.tsv`, `corpus_stats.txt`, `excluded_reviews.tsv` |
| 2. Normalización + vectores | `python main.py build` | `data/opinions_normalized_corpus.tsv`, `models/*.pkl` (12) |
| 3. Consulta | `python main.py search "se traba mucho" --features unigram --rep tfidf` | Top‑10 en pantalla |
| Armar benchmark | `python main.py pool` → etiquetar → `python src/label_pool.py --collect` | `benchmark/qrels.tsv` |
| 4. Benchmark | `python main.py benchmark --all` | `results/rankings.tsv`, métricas |

## Estructura

```
main.py                    # interfaz con los comandos de arriba
config/apps.json           # qué juegos baja cada integrante
src/collect_steam.py       # 1. recolección desde la API pública de Steam
src/merge_corpus.py        #    une, deduplica, filtra y asigna review_id
src/normalize.py           # 2. limpieza, tokenización, POS, stopwords, lematización
src/vectorize.py           #    12 representaciones VSM -> models/*.pkl
src/search.py              # 3. consulta -> coseno -> top-10
src/label_pool.py          #    ayudante para etiquetar qrels
src/benchmark.py           # 4. P@10, R@10, F1@10, MRR, nDCG@10
data/raw/<integrante>/     # JSONL crudos (texto original, sin tocar)
benchmark/                 # queries.tsv, qrels.tsv (hechos a mano)
results/                   # rankings y métricas
```

## Flujo del equipo

1. Cada quien corre `python main.py collect --member <su_nombre>` y sube **solo** `data/raw/<su_nombre>/`. Como nadie escribe en la carpeta de otro, no hay conflictos.
2. Con los 3 listos, una persona corre `python main.py merge`, revisa `corpus_stats.txt` y congela: commit + tag `corpus-v1`.
3. ⚠️ Después de `corpus-v1` **no** se regenera el TSV: los `review_id` son los `document_id` de `qrels.tsv`.

## Fuente y procedimiento (para el reporte)

- API pública de reseñas de Steam: `store.steampowered.com/appreviews/<appid>?json=1` (sin login, sin API key, sin evadir CAPTCHAs).
- Nombre y géneros: `store.steampowered.com/api/appdetails`.
- Idiomas: `latam` (español de Latinoamérica) y `spanish` (español de España).
- Pausa de 1.5 s entre peticiones, reintentos con espera ante errores/429, log de errores en `logs/`.
- Se descartan reseñas de menos de 15 palabras.
- Por juego se bajan 50 % negativas y 50 % positivas (`review_type`), ordenadas por utilidad (`filter=all`, `day_range=365`). Si un juego no tiene suficientes negativas, se completa con positivas.
- Al unir se excluyen reseñas en otro idioma o con arte ASCII (quedan registradas en `data/excluded_reviews.tsv`).
- `rating`: Steam solo da recomendado (1) / no recomendado (0).
- No se guarda el identificador del autor.
