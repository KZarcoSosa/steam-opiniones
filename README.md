# Práctica II — Recuperación de opiniones (PLN)

Corpus de reseñas de **Steam** en español y su normalización con spaCy. Etapa actual: recolección + normalización.

Equipo: Kevin, Angel, Erick. Explicación detallada: [`docs/GUIA.md`](docs/GUIA.md).

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Comandos

| Paso | Comando | Genera |
|---|---|---|
| 1. Recolección | `python main.py collect --member kevin` | `data/raw/kevin/*.jsonl` |
| Unir corpus | `python main.py merge --max-per-game 41` | `data/opinions_raw_corpus.tsv`, `corpus_stats.txt`, `excluded_reviews.tsv` |
| 2. Normalización | `python main.py normalize` | `data/opinions_normalized_corpus.tsv` |
| Demostración | `python main.py demo --id 416` · `--text "..."` · `--stats` | Paso a paso en pantalla |

## Estructura

```
main.py                    # interfaz con los comandos de arriba
config/apps.json           # qué juegos baja cada integrante
src/collect_steam.py       # 1. recolección desde la API pública de Steam
src/merge_corpus.py        #    une, deduplica, filtra y asigna review_id
src/normalize.py           # 2. limpieza, tokenización, POS, stopwords, lematización
src/demo_normalize.py      #    demostración paso a paso
data/raw/<integrante>/     # JSONL crudos (texto original, sin tocar)
data/                      # corpus crudo, normalizado y estadísticas
```

## Corpus congelado

El corpus (`corpus-v1`) tiene 492 reseñas. **No** se vuelve a correr `merge` ni `collect`: los `review_id` se usarán como `document_id` en el benchmark.

## Fuente y procedimiento (para el reporte)

- API pública de reseñas de Steam: `store.steampowered.com/appreviews/<appid>?json=1` (sin login, sin API key, sin evadir CAPTCHAs).
- Nombre y géneros: `store.steampowered.com/api/appdetails`.
- Idiomas: `latam` (español de Latinoamérica) y `spanish` (español de España).
- Pausa de 1.5 s entre peticiones, reintentos con espera ante errores/429, log de errores en `logs/`.
- Se descartan reseñas de menos de 15 palabras.
- Por juego se bajan 50 % negativas y 50 % positivas (`review_type`), ordenadas por utilidad (`filter=all`, `day_range=365`). Si un juego no tiene suficientes negativas, se completa con positivas.
- Al unir se excluyen reseñas en otro idioma o con arte ASCII (registradas en `data/excluded_reviews.tsv`).
- `rating`: Steam solo da recomendado (1) / no recomendado (0).
- No se guarda el identificador del autor.
