# Recuperación de opiniones de Steam — PLN, Práctica II (Etapa 1)

Buscador de opiniones con el **Modelo de Espacio Vectorial (VSM)** y **similitud coseno** sobre **492 reseñas de Steam en español** de **12 videojuegos**. El usuario escribe un aspecto o experiencia en texto libre y el sistema devuelve el **Top-10** de reseñas más parecidas.

**Equipo:** Lecanda Castillo Ángel · Ruiz Martines Erick · Zarco Sosa Kevin

📄 **Reporte:** [`docs/reporte.pdf`](docs/reporte.pdf) 
**Enunciado:** `05_Opinion_Retrieval_Project_Stage_1.pdf`

---

## Dónde está cada entregable (sección 11 — Evidencias)

| Entregables | Archivo / carpeta | Notas |
|---|---|---|
| **Código fuente** | [`main.py`](main.py), [`src/`](src) | Un script por etapa (ver tabla de comandos) |
| **Corpus crudo** (sec. 5) | [`data/opinions_raw_corpus.tsv`](data/opinions_raw_corpus.tsv) | 492 reseñas con todos los campos de la sec. 5 (`review_id`, `item_id`, `item_name`, `review`, `rating`, `date`, `category`, `source`, `url`, …) |
| Respuestas originales de la API | [`data/raw/kevin/`](data/raw/kevin), [`data/raw/angel/`](data/raw/angel), [`data/raw/erick/`](data/raw/erick) | JSONL exactamente como los devolvió Steam, un archivo por juego |
| Estadísticas del corpus | [`data/corpus_stats.txt`](data/corpus_stats.txt) | Distribución por juego, rating, idioma, año, género e integrante |
| Reseñas excluidas | [`data/excluded_reviews.tsv`](data/excluded_reviews.tsv) | 21 reseñas en otro idioma o con arte ASCII, con el motivo |
| **Corpus normalizado** (sec. 6) | [`data/opinions_normalized_corpus.tsv`](data/opinions_normalized_corpus.tsv) | Columnas `review_norm` (solo reseña) e `item_review_norm` (nombre del juego + reseña) |
| **Representaciones vectoriales** (sec. 7) | [`models/v1/`](models/v1) | 12 configuraciones: matriz dispersa `.npz` + vectorizador `.joblib` (equivalente a los PKL) + `manifest.json` |
| **Benchmark: consultas** (sec. 9) | [`data/benchmark/queries.tsv`](data/benchmark/queries.tsv) | 15 consultas: 5 fáciles, 5 medias, 5 difíciles |
| **Benchmark: documentos relevantes** (sec. 9) | [`data/benchmark/qrels.tsv`](data/benchmark/qrels.tsv) | 180 juicios (12 por consulta; relevancia 1 = parcial, 2 = muy relevante) |
| Congelamiento del benchmark | [`data/benchmark/benchmark_lock.json`](data/benchmark/benchmark_lock.json) | Sumas SHA-256 de corpus, consultas y qrels |
| **Rankings del sistema** (sec. 9) | [`data/rankings.tsv`](data/rankings.tsv) | Top-10 por consulta (`query_id`, `document_id`, `rank`, `score`) con review + unigramas + TF-IDF |
| **Resultados por consulta y promedios** | [`data/results/metrics_by_query.tsv`](data/results/metrics_by_query.tsv), [`data/results/metrics_summary.tsv`](data/results/metrics_summary.tsv) | P@10, R@10, F1@10, RR/MRR, nDCG@10, por consulta y por nivel |
| **Salida del evaluador del profesor** | [`data/results/evaluacion_profesor/`](data/results/evaluacion_profesor) | `salida_evaluador.txt` (lo que imprime) y `evaluation_results.tsv`. El script original, sin modificar, está en [`tools/evaluador_profesor/`](tools/evaluador_profesor) |
| Comparación de las 12 configuraciones | [`data/results/all_configurations.tsv`](data/results/all_configurations.tsv) | Métricas promedio de cada combinación |
| **Análisis de errores** (sec. 12) | Sección 9 del reporte y [`docs/analisis_errores.tsv`](docs/analisis_errores.tsv) | 5 consultas con mejor y 5 con peor desempeño, con la causa de cada una (evidencia automática en [`data/results/error_analysis_draft.tsv`](data/results/error_analysis_draft.tsv)) |
| **Documento PDF** (resultados, análisis de errores, integrantes) | **[`docs/reporte.pdf`](docs/reporte.pdf)**  | Portada, corpus, normalización, representaciones, benchmark, resultados, evaluador oficial, comparación de las 12 configuraciones y análisis de errores |
| Explicación detallada del proceso | [`docs/GUIA.md`](docs/GUIA.md) | Recolección y normalización paso a paso |

---

## Resultados (evaluador del profesor)

Configuración: `review` + `unigram` + `tfidf`. Salida completa en [`data/results/evaluacion_profesor/salida_evaluador.txt`](data/results/evaluacion_profesor/salida_evaluador.txt).

| P@10 | R@10 | F1@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| 0.2733 | 0.2278 | 0.2485 | 0.5873 | 0.2740 |

| Nivel | P@10 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| 1 Fácil | 0.440 | 0.367 | 0.695 | 0.410 |
| 2 Media | 0.240 | 0.200 | 0.567 | 0.259 |
| 3 Difícil | 0.140 | 0.117 | 0.500 | 0.153 |

- **Mejores consultas (nDCG@10):** Q01, Q03, Q09, Q02 y Q08.
- **Peores consultas (nDCG@10):** Q11, Q13, Q06, Q07 y Q12.

`python main.py casos` imprime esta lista con el texto y las métricas de cada consulta.

---

## Cómo se cumple cada sección del documento

| Sección | Qué se hizo | Dónde |
|---|---|---|
| **4. Recolección** | API pública de Steam (`store.steampowered.com/appreviews/<appid>?json=1`), sin login ni API key. Idiomas `latam` y `spanish`. Se piden negativas y positivas por separado para equilibrar el rating. Mínimo 15 palabras por reseña. Pausa de 1.5 s entre peticiones, reintentos ante errores/429 y log de errores. No se guarda el identificador del autor. | `src/collect_steam.py`, `config/apps.json` |
| **5. Corpus crudo** | Unión de los tres integrantes, sin duplicados, filtro de idioma, 41 reseñas por juego, `review_id` consecutivo | `src/merge_corpus.py` |
| **6. Normalización** | spaCy `es_core_news_sm`: limpieza (BBCode, URLs, caracteres de control), tokenización, POS, eliminación de artículos, preposiciones, conjunciones y pronombres, lematización, minúsculas y sin acentos. Se conservan "no", "muy" y "nunca". | `src/normalize.py` |
| **7. Representación** | 2 contenidos (`review`, `item_review`) × 2 características (unigramas, bigramas) × 3 pesos (frecuencia, binaria, TF-IDF) = 12 modelos | `src/vectorize.py` |
| **8. Consulta** | La consulta se normaliza con **la misma función** que el corpus, se vectoriza y se compara por coseno. Se muestra el Top-10. | `src/retrieve.py` |
| **9. Benchmark** | 15 consultas en 3 niveles con sus juicios; rankings y métricas | `src/benchmark.py`, `data/benchmark/` |
| **10. Interfaz** | Comandos separados en `main.py` | ver abajo |
| **11. Evidencias** | Tabla de arriba | |
| **12. Análisis de errores** | 5 mejores y 5 peores consultas con sus causas | `docs/reporte.pdf` (sección 9), `docs/analisis_errores.tsv` |
| **13. Criterio de término** | Se puede actualizar el corpus, generar representaciones, consultar en texto libre, ver el Top-10 y medir el benchmark. El `rating` se conserva para etapas posteriores. | |

---

## Instalación

Python 3.10 o posterior. En Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

En Linux/macOS se activa con `source .venv/bin/activate`. El modelo `es_core_news_sm` es obligatorio para normalizar las consultas igual que el corpus.

## Comandos (sección 10 — Interfaz)

| Tarea | Comando |
|---|---|
| Recolectar / actualizar el corpus ⚠ | `python main.py collect --member kevin` |
| Unir el corpus ⚠ | `python main.py merge --max-per-game 41` |
| Normalizar | `python main.py normalize` |
| Ver la normalización paso a paso | `python main.py demo --text "mala optimización y bajos FPS"` |
| Generar las 12 representaciones | `python main.py vectorize` |
| **Consultar (Top-10)** | `python main.py search --query "el juego se traba y baja de FPS" --content review --features unigram --weight tfidf` |
| Correr las 5 consultas de un nivel del benchmark | `python main.py search --nivel 1` (también `--qid Q02`) |
| Validar el benchmark | `python main.py benchmark --validate` |
| **Correr el benchmark y las métricas** | `python main.py benchmark` (`--all` compara las 12 configuraciones) |
| Evaluador del profesor | `python tools/evaluador_profesor/evaluate_ir_metrics.py data/rankings.tsv data/benchmark/qrels.tsv --output data/results/evaluacion_profesor/evaluation_results.tsv` |
| Mejores y peores consultas | `python main.py casos` (también `--nivel 2` o `--por-nivel`) |
| Pruebas automáticas | `python -m pytest -q tests/test_retrieval.py` |

Opciones de búsqueda: `--content review|item_review`, `--features unigram|bigram`, `--weight frequency|binary|tfidf`.

⚠ `collect` y `merge` cambian los `review_id`, y los juicios de `qrels.tsv` dejarían de coincidir. No hay que correrlos para revisar la entrega.

## Estructura

```text
main.py                              Interfaz de comandos
config/apps.json                     Juegos que bajó cada integrante
src/collect_steam.py                 Recolección (API pública de Steam)
src/merge_corpus.py                  Unión, filtro y estadísticas
src/normalize.py                     Normalización con spaCy
src/demo_normalize.py                Demostración paso a paso de la normalización
src/vectorize.py                     12 modelos VSM
src/retrieve.py                      Consulta, coseno y Top-10
src/benchmark.py                     Validación, rankings y métricas
src/error_analysis.py                Insumos para el análisis de errores
src/best_worst.py                    Mejores y peores consultas
data/raw/<integrante>/               JSONL originales de Steam
data/opinions_raw_corpus.tsv         Corpus crudo
data/opinions_normalized_corpus.tsv  Corpus normalizado
data/benchmark/                      queries.tsv, qrels.tsv, auditoría y lock
data/rankings.tsv                    Top-10 por consulta
data/results/                        Métricas y salida del evaluador del profesor
models/v1/                           Matrices .npz, vectorizadores .joblib, manifest
tools/evaluador_profesor/            Script de evaluación del curso (sin modificar)
docs/reporte.pdf                     Reporte final (PDF)
docs/reporte_etapa1.docx             Reporte en Word (editable)
docs/GUIA.md                         Explicación detallada del proceso
tests/test_retrieval.py              Pruebas
```
