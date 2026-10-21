# Auditoría del proyecto — Etapa 1 (PDF del profesor)

Fecha de revisión: 21 de octubre de 2026. Base: contenido de `steam-opiniones-main.zip`, frente a los 13 apartados del documento de especificaciones. Se conserva el corpus congelado `corpus-v1`.

| Apartado | Estado al recibir ZIP | Trabajo verificado/agregado | Estado actual |
|---|---|---|---|
| 1. Introducción | Explicada | La fuente menciona búsqueda por aspecto, no clasificación de sentimiento | Cubierto |
| 2. Objetivo | Definido | Buscador VSM implementado en `src/retrieve.py` | Código completo; prueba real con spaCy pendiente |
| 3. Generales | Parcial | Tres ponderaciones, n-gramas, coseno, benchmark, código | Prueba end-to-end pendiente |
| 4. Recolección | Implementada | API pública Steam, JSONL original, distribuido en 3 integrantes | Cubierto; no recolectado de nuevo |
| 5. Corpus crudo | Completo | `data/opinions_raw_corpus.tsv`: 492 filas, 15 campos, `review_id` únicos | Verificado |
| 6. Normalización | Implementada | `data/opinions_normalized_corpus.tsv`: 492 filas, `review_norm`, `item_review_norm` | Verificado sobre archivos; no reejecutado por falta de modelo spaCy |
| 7. VSM | Faltaba | `src/vectorize.py`, 12 pares vectorizador/matriz | Implementado y probado (15 tests) |
| 8. Recuperación | Faltaba | `src/retrieve.py`: consulta libre, normalización idéntica, coseno, Top-10, score | Implementado; falta prueba de texto libre con spaCy instalado |
| 9. Benchmark | Faltaba | 15 consultas, 180 etiquetas iniciales, programa de P@10, R@10, F1@10, MRR y nDCG@10 | Código y estructura verificados; etiquetas por validar y métricas reales sin ejecutar |
| 10. Interfaz | Parcial | `main.py` con `collect`, `merge`, `normalize`, `vectorize`, `search`, `benchmark`, `report`, `demo` | Código completo; requiere spaCy para ejecución íntegra |
| 11. Evidencias | Parcial | Código, TSV, modelos, qrels, reporte preliminar PDF, tests | Pendientes rankings, métricas medidas y reporte final |
| 12. Análisis de errores | Faltaba | Script de 5 mejores y 5 peores con candidatos de causas y evidencia de omisiones | Pendiente ejecutar benchmark y explicar/confirmar causas reales |
| 13. Cierre | No | Flujo de actualización, representación y búsqueda implementado | **No alcanzado** hasta verificar consulta real y métricas de benchmark |

## Estadísticas comprobadas

- 492 reseñas únicas, todas con texto, `review_id` y metadatos requeridos.
- 12 juegos: 41 opiniones por juego. Rating: 255 recomendado y 237 no recomendado.
- 15 consultas (5 por nivel) y 180 juicios propuestos (12 por consulta; grados 1 y 2).
- 12 configuraciones VSM generadas. Para reseñas solas: 5,473 unigramas y 21,647 bigramas; para nombre + reseña: 5,472 unigramas y 21,978 bigramas.
- Se verificó el cálculo con el ejemplo del PDF: P@10=0.50; R@10=0.833; F1@10=0.625; RR=1.0; nDCG@10≈0.835.

## Pendientes imprescindibles antes de declarar terminado el proyecto

1. **Revisar individualmente los 180 juicios iniciales** de relevancia. No declarar que se realizó anotación humana mientras `human_checked=no`.
2. **Instalar el modelo** `python -m spacy download es_core_news_sm` en una computadora con Internet, para normalizar las consultas exactamente como las reseñas.
3. **Evaluar:** `python main.py benchmark --approve` (tras validación), luego `python main.py benchmark --all`. Comprobar `data/rankings.tsv`, `data/results/metrics_by_query.tsv`, `metrics_summary.tsv`, `all_configurations.tsv`.
4. **Análisis de errores:** revisar los 5 mejores y 5 peores según los resultados observados. En `error_analysis_draft.tsv`, completar y verificar `team_analysis` por consulta. Discutir sinónimos, poco solapamiento léxico, términos ambiguos y errores de normalización según evidencias.
5. **Regenerar PDF:** `python main.py report`. Confirmar que ya contiene métricas reales y explicaciones específicas y que los nombres del equipo están bien escritos.

## Cuestiones metodológicas por defender

- Unigrama y bigrama se modelan **por separado** (`ngram_range=(1,1)` frente a `(2,2)`); no se trata de 1+2 conjuntamente, aunque se puede justificar extenderlo para otros experimentos.
- El rating de Steam es **binario** (recomendado/no recomendado), no una escala de 1–5. Se guarda pero **no se usa** como señal de relevancia.
- En los cómputos de precisión/recall, un documento no listado entre los positivos del benchmark se toma como no relevante. Los 10–15 juicios positivos solicitados por consulta **no son necesariamente exhaustivos**: hay un sesgo potencial que debe reconocerse.
- El MRR se obtiene usando el **ranking completo**; la interfaz/archivo `rankings.tsv` guarda y presenta solamente los 10 primeros, como pide el documento.
- El informe de avance actual no presenta resultados inventados: se actualizará cuando `benchmark` se ejecute con el modelo correspondiente.
