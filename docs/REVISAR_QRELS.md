# Revisión humana requerida antes de congelar el benchmark

El archivo `data/benchmark/annotation_audit.tsv` incluye `query_id`, `document_id`, `relevance`, `item_name`, el inicio de la reseña original, y `human_checked=no`.

1. Abrir el TSV en editor de tablas y leer el texto de cada juicio. Si el fragmento no es suficiente, buscar el `review_id` en `data/opinions_raw_corpus.tsv` y leer **la reseña completa**.
2. Para cada consulta valorar exclusivamente el **tema o experiencia**, no si el usuario recomienda el juego ni el rating de Steam. Usar 2 cuando la reseña aborda clara y sustancialmente el tema y 1 cuando lo aborda parcialmente. Si es irrelevante, sustituirla en `qrels.tsv` por una reseña realmente relevante; actualizar la fila correspondiente en la auditoría.
3. Mantener 10–15 documentos con relevancias 1 y 2 por consulta, incluyendo al menos uno de cada grado. No incluir ceros en `qrels.tsv`.
4. **Únicamente tras verificar**, cambiar `human_checked` a `yes` en cada fila y ejecutar `python main.py benchmark --approve`. Este comando exige coincidencia de IDs, grados y número de anotaciones y escribe las sumas SHA-256 finales.
5. Luego ejecutar `python main.py benchmark --all`, revisar los casos 5 mejores y 5 peores y explicar causas observadas.

La auditoría inicial fue asistida por IA y es útil como propuesta, pero **no debe presentarse al profesor como etiquetas manualmente revisadas** hasta que el equipo las compruebe.
