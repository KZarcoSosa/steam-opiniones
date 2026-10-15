# Guía del proyecto — Práctica II: Recuperación de opiniones

## 1. ¿De qué va la práctica?

Construir un **buscador de reseñas**. El usuario escribe un problema o experiencia en texto libre ("el juego se traba mucho", "la historia es aburrida") y el programa devuelve las **10 reseñas más parecidas**, ordenadas por similitud.

No es análisis de sentimiento: el rating se guarda para después, pero aquí solo importa **encontrar las opiniones que hablan de lo que se buscó**.

El "parecido" se mide con el **Modelo de Espacio Vectorial (VSM)**. Cada reseña es un vector con una dimensión por cada palabra (o par de palabras) del vocabulario. La consulta se convierte en un vector del mismo espacio y se compara con todas las reseñas usando la **similitud coseno**.

### Entregables (sección 11)

| Entregable | Archivo | Estado |
|---|---|---|
| Código fuente | `src/`, `main.py` | ✅ completo |
| Corpus crudo | `data/opinions_raw_corpus.tsv` | ⏳ faltan las reseñas de Angel y Erick |
| Corpus normalizado | `data/opinions_normalized_corpus.tsv` | ✅ script listo |
| Representaciones | `models/*.pkl` (12) | ✅ script listo |
| Benchmark | `benchmark/queries.tsv`, `qrels.tsv` | ❌ hay que escribirlo y etiquetarlo a mano |
| Resultados | `results/rankings.tsv` + métricas | ✅ script listo (falta el benchmark) |
| Reporte PDF | resultados, análisis de errores, nombres | ❌ |

---

## 2. Recolección (cómo se obtuvieron las reseñas)

**Archivo:** `src/collect_steam.py` · **Comando:** `python main.py collect --member kevin`

### 2.1 La fuente

Steam publica las reseñas de cada juego en una URL pública que regresa JSON. No pide login ni API key:

```
https://store.steampowered.com/appreviews/<appid>?json=1
```

`appid` es el número que aparece en la URL de la tienda. Por ejemplo, `store.steampowered.com/app/1091500/` es Cyberpunk 2077. Para el nombre y los géneros del juego se usa otra URL pública: `store.steampowered.com/api/appdetails?appids=<appid>`.

### 2.2 Parámetros que mandamos

| Parámetro | Valor | Para qué |
|---|---|---|
| `language` | `latam` y luego `spanish` | Español de Latinoamérica y de España |
| `review_type` | `negative` y luego `positive` | Pedirlas por separado para lograr 50/50 |
| `filter` | `all` | Ordenadas por "más útiles", no por más recientes |
| `day_range` | `365` | Sin esto, `filter=all` solo busca en los últimos días |
| `num_per_page` | `100` | Máximo por página |
| `cursor` | `*`, luego el que regresa Steam | Paginación: cada respuesta trae el cursor de la siguiente página |
| `purchase_type` | `all` | Incluye reseñas de quien no compró en Steam |

### 2.3 El algoritmo, paso a paso

```
para cada juego asignado al integrante (config/apps.json):
    pedir nombre y géneros (appdetails)
    meta: 30 negativas + 30 positivas
    para tipo en [negative, positive]:
        para idioma en [latam, spanish]:
            cursor = "*"
            mientras falten reseñas de ese tipo:
                pedir página (100 reseñas) con ese cursor
                para cada reseña:
                    descartar si ya la tenemos (recommendationid repetido)
                    descartar si tiene menos de 15 palabras
                    guardar como una línea JSON en data/raw/<integrante>/<appid>.jsonl
                si el cursor no cambió o no hubo reseñas: se acabaron -> salir
                esperar 1.5 s
    si faltaron negativas: completar con positivas (y viceversa)
```

### 2.4 Decisiones y su porqué (útil para el reporte)

- **50 % negativas.** La primera prueba dio 209 positivas de 240. Con eso no hay suficientes reseñas sobre problemas (bugs, optimización) para el benchmark.
- **`filter=all` + `day_range`.** Con `filter=recent` todo salía de 2026. Ahora las fechas van de 2019 a 2026, y la práctica pide reportar la distribución por fecha.
- **Mínimo 15 palabras.** Un "10/10" o un "buen juego" no tiene nada que recuperar.
- **Buenas prácticas** (las pide la sección 4): pausa de 1.5 s, reintentos con espera creciente, espera de 30 s si Steam responde 429 (demasiadas peticiones) y log de errores en `logs/`. No se guarda el `steamid` del autor.
- **Versión cruda + estructurada.** El JSONL guarda el texto exacto de Steam y el TSV es la versión estructurada.
- **Re-ejecutable.** Si se corre dos veces no duplica (revisa los `recommendationid` existentes). `--overwrite` empieza de cero.

### 2.5 Unión del corpus

**Archivo:** `src/merge_corpus.py` · **Comando:** `python main.py merge`

1. Lee todos los `data/raw/*/*.jsonl`.
2. Quita duplicados por `recommendationid`.
3. Excluye reseñas que no están en español o que son arte ASCII. Steam etiqueta el idioma según la cuenta del usuario, no según el texto. Las excluidas quedan en `excluded_reviews.tsv`; en los datos de Kevin fueron 11 de 240.
4. Ordena por juego e id y asigna `review_id` = 1, 2, 3… (este es el `document_id` de los qrels).
5. Escribe `opinions_raw_corpus.tsv` con los campos de la sección 5, más `corpus_stats.txt` con la distribución por juego, rating, año, idioma y género.

---

## 3. Normalización

**Archivo:** `src/normalize.py` · **Comando:** `python main.py build`

Ejemplo real:

> **Original:** El juego es muy bonito y lindo, pero hay un grave problema, para los jugadores de bajos recursos no lo recomiendo hay muchos bugs visuales y muy mala optimizacion
>
> **Normalizada:** juego ser mucho bonito lindo haber grave problema jugador bajo recurso no recomeir haber mucho bugs visual mucho malo optimizacion

| Paso | Qué hace | Ejemplo |
|---|---|---|
| Limpieza | Quita BBCode de Steam, URLs, caracteres de control y espacios repetidos | `[h1]Malo[/h1]` → `Malo` |
| Tokenización | spaCy (`es_core_news_sm`) parte el texto en tokens | |
| POS + stopwords | Quita **artículos** (DET Art), **preposiciones** (ADP), **conjunciones** (CCONJ/SCONJ) y **pronombres** (PRON, DET posesivos) | el, de, y, pero, lo, mi |
| Protección | Solo quita un token de esas categorías si además es una palabra funcional conocida | spaCy etiqueta "fps" como preposición; así no se pierde |
| Lematización | Forma base, en minúsculas | jugadores → jugador, es → ser |
| Acentos | Se quitan (la ñ se conserva) | optimización = optimizacion |

**Se conservan** adverbios como "no", "muy" y "nunca" porque cambian el sentido de la opinión. También se quitan signos de puntuación y números sueltos.

**Para el análisis de errores:** el lematizador falla con algunos verbos ("recomiendo" → "recomeir", "traba" → "trar") y a veces etiqueta mal palabras al inicio de la frase ("Muchos bugs" → "muchos" como pronombre).

---

## 4. Representación vectorial

**Archivo:** `src/vectorize.py`

Se generan **12 combinaciones** y cada una queda en `models/<contenido>_<rep>_<ngrama>.pkl`:

| Dimensión | Opciones |
|---|---|
| Contenido | `review` (solo texto) · `item_review` (nombre del juego + texto) |
| Características | `unigram` (palabras sueltas) · `bigram` (pares de palabras seguidas) |
| Representación | `frequency` (cuántas veces) · `binary` (aparece sí/no) · `tfidf` (frecuencia × rareza) |

Con las 229 reseñas de Kevin: unigramas ≈ 3,700 términos y bigramas ≈ 13,000.

**TF-IDF:** una palabra pesa más si aparece mucho en esa reseña pero en pocas reseñas del corpus. "juego" aparece en casi todas, así que pesa poco; "crashea" pesa mucho.

**Similitud coseno:** `cos(q, d) = (q · d) / (|q| · |d|)`. Va de 0 (no comparten términos) a 1 (misma dirección).

---

## 5. Búsqueda

**Archivo:** `src/search.py`

```
python main.py search "se traba mucho y tiene muchos bugs" --content review --features unigram --rep tfidf
```

1. Normaliza la consulta con **la misma función** que el corpus.
2. `vectorizer.transform()` la pone en el mismo espacio.
3. Calcula el coseno contra todas las reseñas.
4. Ordena de mayor a menor y muestra el top-10 con su score.

---

## 6. Benchmark (lo que falta y se hace a mano)

### 6.1 Escribir 15 consultas en `benchmark/queries.tsv`

5 fáciles (nivel 1), 5 medias (nivel 2) y 5 difíciles (nivel 3). Temas que sí aparecen en el corpus de Kevin, con el número de reseñas que los mencionan:

| Tema | Reseñas | Ideas de consulta |
|---|---|---|
| bugs | 35 | 1: "muchos bugs" · 2: "el juego tiene errores y fallos" · 3: "nada funciona como debería" |
| optimización / rendimiento / fps | 23 / 18 / 11 | 1: "mala optimización" · 2: "corre lento en mi computadora" · 3: "mi PC no aguanta el juego" |
| historia | 46 | 1: "buena historia" · 2: "la trama engancha" · 3: "me hizo llorar el final" |
| aburrido | 17 | 1: "juego aburrido" · 2: "se vuelve repetitivo" · 3: "a las pocas horas ya no quería jugar" |
| DLC / precio | 18 / 8 | 1: "demasiados DLC" · 2: "muy caro para lo que ofrece" · 3: "sacan dinero con contenido que debería venir incluido" |
| crashes | 10 | 1: "el juego crashea" · 2: "se cierra solo" · 3: "pierdo mi partida a cada rato" |
| dificultad | 10 | 1: "juego difícil" · 2: "curva de aprendizaje alta" · 3: "no entendía qué hacer al principio" |

Cuando entren los juegos de Angel (CS2, Elden Ring, Hollow Knight, Valheim) y Erick (Hades, Terraria, Wukong, Dota 2) aparecerán más temas: tramposos, servidores, multijugador, jefes, comunidad tóxica.

### 6.2 Etiquetar 10-15 relevantes por consulta (`qrels.tsv`)

1. `python main.py pool` crea `benchmark/pool/Q01.tsv`… con candidatos.
2. Se abren en Excel y se pone `relevance` = 2 (muy relevante), 1 (parcial) o 0 (no relevante).
3. En las consultas difíciles el sistema casi no encuentra nada (es justo lo que se quiere medir), así que hay que buscar a mano en el corpus con Ctrl+F y sinónimos.
4. `python src/label_pool.py --collect` genera `qrels.tsv` y avisa si alguna consulta no tiene entre 10 y 15 relevantes.

### 6.3 Evaluar

`python main.py benchmark --all` evalúa las 12 configuraciones, dice cuál es la mejor y genera `rankings.tsv` con ella.

| Métrica | Qué mide |
|---|---|
| P@10 | De los 10 resultados, cuántos son relevantes |
| R@10 | De todos los relevantes, cuántos salieron en el top-10 |
| F1@10 | Balance entre P y R |
| MRR | Qué tan arriba sale el primer relevante (1, 1/2, 1/3…) |
| nDCG@10 | Calidad del orden: premia que los "2" salgan antes que los "1" |

**Resultado esperado** (para el análisis de errores): las fáciles van bien; las difíciles van mal porque el VSM solo cuenta palabras iguales y no entiende sinónimos. "Mi PC no aguanta" no comparte ninguna palabra con "mala optimización".

---

## 7. Preguntas para repasar

1. ¿Por qué la consulta se normaliza con la misma función que el corpus? ¿Qué pasaría si no?
2. ¿Por qué bigramas da menos resultados que unigramas en consultas cortas?
3. ¿En qué caso binary podría ganarle a TF-IDF?
4. ¿Por qué `item_review` podría ayudar en "Cyberpunk tiene muchos bugs" pero empeorar en "muchos bugs"?
5. ¿Por qué no se quita "no" como stopword?
6. ¿Por qué el `review_id` no puede cambiar después de etiquetar los qrels?
