# Recuperación de opiniones de Steam — PLN, Etapa 1

Buscador VSM de **492 reseñas de Steam en español** sobre **12 videojuegos**; equipo: Kevin, Ángel y Erick. Este repositorio mantiene las fuentes y los identificadores originales y agrega la recuperación y evaluación solicitadas en `05_Opinion_Retrieval_Project_Stage_1.pdf`.

## Requisitos

Python 3.10 o posterior. En Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

En Linux/macOS, activa con `source .venv/bin/activate` en vez de `Activate.ps1`.

**El modelo de español `es_core_news_sm` es obligatorio** para consultas y evaluación. Los modelos VSM ya están incluidos en `models/v1/` y se pueden examinar sin descargarlo; sin el modelo spaCy no se pueden verificar consultas nuevas con la misma normalización que el corpus. No reemplazarlo con otro normalizador silenciosamente.

## Uso por secciones del proyecto

| Sección | Comando | Resultado |
|---|---|---|
| 4 | `python main.py collect --member kevin` | Recopilar reseñas (⚠ cambia datos) |
| 5 | `python main.py merge --max-per-game 41` | Regenerar corpus (⚠ cambia review_id) |
| 6 | `python main.py normalize` | Normalizar reseñas usando spaCy |
| 6 | `python main.py demo --id 416` | Explicar paso a paso la normalización |
| 7 | `python main.py vectorize` | Guardar **12** modelos (2 campos × 2 n-gramas × 3 ponderaciones) |
| 8 | `python main.py search --query "el juego se traba y baja de FPS" --content review --features unigram --weight tfidf` | Mostrar **Top-10** y similitud coseno |
| 9 | `python main.py benchmark --validate` | Validar estructura e integridad de 15 consultas y 180 juicios |
| 9 | `python main.py benchmark` | Crear `data/rankings.tsv`, métricas individuales y promedio |
| 9 | `python main.py benchmark --all` | Comparar las 12 configuraciones además de generar el ranking |
| 11-12 | `python main.py report` | Actualizar `docs/reporte_etapa1.pdf` |

Opciones de búsqueda: `--content review|item_review`; `--features unigram|bigram`; `--weight frequency|binary|tfidf`. Todos los modelos están almacenados en matrices comprimidas (`.npz`) y vectorizadores (`.joblib`). **Se aplica `src.normalize.normalize_text` tanto a las reseñas como a las consultas.**

## Antes de reportar resultados de evaluación como definitivos

El benchmark nuevo (`data/benchmark/queries.tsv`, `qrels.tsv`) contiene **15 consultas**: 5 fáciles, 5 medias y 5 difíciles, con 12 documentos inicialmente relevantes por consulta. Las etiquetas 1 y 2 son una **propuesta de anotación asistida por IA, NO relevancias humanas verificadas**. La evaluación funciona con ellas como diagnóstico, pero el equipo debe:

1. Leer las **180** filas de `data/benchmark/annotation_audit.tsv`, contrastar las reseñas originales, y corregir en `qrels.tsv` y `annotation_audit.tsv` cada juicio que lo necesite. Para las que ya estén correctas, establecer `human_checked` en `yes` (todas las filas).
2. Ejecutar `python main.py benchmark --approve` para cotejar la auditoría y fijar las sumas SHA-256 de los archivos ya revisados. No marcar como revisadas etiquetas que no se hayan inspeccionado.
3. Ejecutar `python main.py benchmark --all` (con spaCy instalado). Examinar `data/results/error_analysis_draft.tsv` y escribir explicaciones de las **5 mejores** y las **5 peores** consultas en `team_analysis`; corroborar las hipótesis automáticas.
4. Ejecutar `python main.py report` para regenerar el PDF con métricas auténticas. Revisar que las explicaciones de errores y los nombres sean adecuados para la entrega.

**Precaución metodológica:** el PDF pide 10–15 positivos por consulta; al evaluar, los documentos sin juicio positivo se consideran no relevantes, aunque algunos podrían serlo. Esto sesga la evaluación si el conjunto no es exhaustivo. El archivo congelado sólo fija los juicios existentes, no demuestra exhaustividad. El resultado de la comparación de configuraciones sobre ese mismo benchmark es exploratorio, no una prueba independiente.

## Estructura principal

```text
main.py                          Comandos independientes
config/apps.json                 Juegos por integrante
src/collect_steam.py             Recolección con API pública
src/merge_corpus.py              Unión, selección y estadísticas
src/normalize.py                 spaCy y POS (el normalizador oficial)
src/vectorize.py                 12 modelos VSM
src/retrieve.py                  Recuperación Top-10 y similitud coseno
src/benchmark.py                 Validación, rankings y métricas
src/error_analysis.py            Evidencia de 5 mejores y 5 peores, requiere revisión
src/report.py                    Reporte en PDF
models/v1/                       Matrices .npz, vectorizadores .joblib, manifiesto
/data/opinions_raw_corpus.tsv    492 reseñas crudas con metadatos
/data/opinions_normalized_corpus.tsv 492 reseñas normalizadas
/data/benchmark/                 queries.tsv, qrels.tsv, auditoría y lock
/data/results/                   Métricas y comparaciones tras ejecutar benchmark
docs/reporte_etapa1.pdf          Informe de avance (se completa con benchmark)
docs/ANALISIS_CUMPLIMIENTO.md    Auditoría de los 13 requisitos
tests/test_retrieval.py          Pruebas de modelos y métricas
```

**No ejecutar `collect` ni `merge` para el benchmark congelado**: alteran los `review_id` y obligan a rehacer los juicios de relevancia. Si realmente se actualiza el corpus, repetir normalización, modelos, anotación y evaluación.

## Verificación automatizada

```bash
python -m pytest -q tests/test_retrieval.py
```

15 pruebas independientes del modelo español verifican las 12 matrices, el ranking, la estructura de archivos y las métricas contra el ejemplo numérico del PDF.

## Fuente y normalización original

Para explicación detallada de adquisición, cuotas, filtros, idioma, POS y lematización, consultar `docs/GUIA.md` (escrito antes de los apartados 7–13). Se conserva íntegra la documentación original y sus datos.
