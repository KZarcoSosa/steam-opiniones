# Recuperación de opiniones de Steam — PLN, Práctica II (Etapas I y II)

Buscador de opiniones con el **Modelo de Espacio Vectorial (VSM)** y **similitud coseno** sobre **492 reseñas de Steam en español** de **12 videojuegos**. El usuario escribe un aspecto o experiencia en texto libre y el sistema devuelve el **Top-10** de reseñas más parecidas.

**Equipo:** Lecanda Castillo Ángel · Ruiz Martinez Erick · Zarco Sosa Kevin

📄 **Reporte Etapa I:** [`docs/reporte.pdf`](docs/reporte.pdf)  
📄 **Reporte Etapa II:** [`docs/reporte_etapa2.pdf`](docs/reporte_etapa2.pdf)   
 La documentación de la Etapa II (análisis de sentimiento) está en la [última sección](#etapa-ii--análisis-de-sentimiento-supervisado) de este README.


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

---

# Etapa II — Análisis de sentimiento supervisado

Clasificación de cada reseña como **positive** o **negative** con aprendizaje automático tradicional (scikit-learn), sin LLMs ni modelos generativos. Se reutilizó el mismo corpus de 492 reseñas, con los mismos `review_id` y sin modificar el benchmark de la Etapa I. El reporte completo está en [`docs/reporte_etapa2.pdf`](docs/reporte_etapa2.pdf).

## Resultado

| | Modelo | Configuración | Macro-F1 |
|---|---|---|---|
| Selección en dev (C014) | LinearSVC | Tokenización y minúsculas, representación binaria (1,2), `C=0.1` | 0.7432 |
| **Test (evaluación única, reentrenado con train+dev)** | LinearSVC | Igual | **0.7421** |

En dev, MultinomialNB (C030, lemas + negación) obtuvo 0.7432 y LogisticRegression (C008) 0.7428. Se registraron 58 experimentos.

## Dónde está cada entregable (sección 11 de la Etapa II)

| Entregable | Archivo | Notas |
|---|---|---|
| **Código con Pipelines** | [`src/sentiment_labels.py`](src/sentiment_labels.py), [`src/sentiment_features.py`](src/sentiment_features.py), [`src/sentiment_experiments.py`](src/sentiment_experiments.py), [`src/sentiment_final.py`](src/sentiment_final.py), [`src/sentiment_incremental.py`](src/sentiment_incremental.py), [`src/sentiment_app.py`](src/sentiment_app.py) | Etiquetas y particiones · preprocesamiento y léxicos · experimentos · evaluación final · impacto incremental · interfaz |
| Entorno, semillas y versiones | [`requirements.txt`](requirements.txt), [`data/stage2/environment.json`](data/stage2/environment.json), [`data/stage2/split_report.txt`](data/stage2/split_report.txt) | Semilla 0 en particiones, validación cruzada y modelos |
| **`labeled_documents.tsv`** | [`data/stage2/labeled_documents.tsv`](data/stage2/labeled_documents.tsv) | `document_id, review, text, rating, label, split` + metadatos. `rating` 1 = positive, 0 = negative |
| Revisión manual de etiquetas | [`data/stage2/label_audit_sample.tsv`](data/stage2/label_audit_sample.tsv) | 50 reseñas: 30 con señales contradictorias y 20 de control |
| **`experiments.tsv`** | [`data/stage2/experiments.tsv`](data/stage2/experiments.tsv) | Bitácora completa: normalización, representación, mejora, desbalance, clasificador, hiperparámetros, macro-F1 en CV y en dev, decisión |
| **`classification_predictions.tsv`** | [`data/stage2/classification_predictions.tsv`](data/stage2/classification_predictions.tsv) | Dev: mejor configuración de cada modelo. Test: `experiment_id = test` |
| **`classification_metrics.tsv`** | [`data/stage2/classification_metrics.tsv`](data/stage2/classification_metrics.tsv) | Accuracy, macro P/R/F1, métricas por clase y matriz de confusión |
| Reportes por clase y matrices de confusión | [`data/stage2/dev_reports.txt`](data/stage2/dev_reports.txt), [`data/stage2/test_report.txt`](data/stage2/test_report.txt) | |
| Errores de dev | [`data/stage2/dev_errors.tsv`](data/stage2/dev_errors.tsv) | El análisis de los 19 errores del modelo seleccionado está en la sección 10 del reporte |
| Impacto incremental de cada preprocesamiento | [`data/stage2/incremental_ablation.tsv`](data/stage2/incremental_ablation.tsv), [`data/stage2/leave_one_out.tsv`](data/stage2/leave_one_out.tsv), [`data/stage2/incremental_summary.txt`](data/stage2/incremental_summary.txt) | Cadena acumulativa con prueba t corregida (Nadeau y Bengio) y ablación "dejar uno fuera" |
| Corpus después de cada preprocesamiento | [`data/stage2/variants/`](data/stage2/variants) | `INDEX.tsv` y `examples.txt` |
| Recursos lingüísticos | [`resources/lexicons/`](resources/lexicons) y sección 6 del reporte | SEL, LIWC 2007 (no se publica por su licencia), léxico de emojis, negación, ortografía y repeticiones |
| Modelo final | [`models/sentiment/final_pipeline.joblib`](models/sentiment/final_pipeline.joblib), [`models/sentiment/final_config.json`](models/sentiment/final_config.json) | Reentrenado con train+dev |
| Sentimiento predicho del corpus | [`data/stage2/predicted_sentiment.tsv`](data/stage2/predicted_sentiment.tsv) | `in_sample = no` solo en las reseñas de test |
| **Reporte** | [`docs/reporte_etapa2.pdf`](docs/reporte_etapa2.pdf), [`docs/reporte_etapa2.docx`](docs/reporte_etapa2.docx) | |

## Comandos de la Etapa II

| Tarea | Comando |
|---|---|
| Etiquetas y particiones 70/15/15 (ya generadas) | `python main.py labels` |
| Ver los preprocesamientos y rasgos de léxico de un texto | `python main.py features --text "No es nada bueno :("` |
| Experimentos en train y dev (ya ejecutados) | `python main.py experiments` |
| Evaluación única en test (ya ejecutada; queda bloqueada) | `python main.py final-test` |
| Impacto incremental de cada preprocesamiento | `python main.py incremental` |
| **Buscar mostrando el sentimiento predicho** | `python main.py search --query "mala optimización y bajos FPS"` |
| **Buscar solo opiniones negativas o positivas** | `python main.py search --query "el juego tiene muchos errores" --sentimiento negativo` |
| Predecir el sentimiento del corpus o de una reseña nueva | `python main.py sentiment predict` · `python main.py sentiment predict --text "No lo recomiendo"` |
| Métricas por clase y matrices de confusión | `python main.py sentiment metrics` |
| Bitácora de experimentos | `python main.py sentiment experiments --stage B --top 15` |
| Errores representativos | `python main.py sentiment errors --split dev -n 10` |
| Pruebas | `python -m pytest -q tests` |

`labels`, `experiments` y `final-test` se niegan a repetirse para proteger el protocolo (particiones fijas y test usado una sola vez). `resources/lexicons/LIWC2007.dic` no se incluye en el repositorio por su licencia.
