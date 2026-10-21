# Nota de actualización (21/10/2026)

Esta guía corresponde a la versión original del proyecto (apartados 4–6). Las secciones 7–13 fueron implementadas posteriormente; consulta `../README.md` y `ANALISIS_CUMPLIMIENTO.md` para su estado actualizado y comandos.

---

# Guía del proyecto — Práctica II: Recuperación de opiniones

## 1. ¿De qué va la práctica?

El proyecto completo es un **buscador de reseñas**. El usuario escribe un problema o experiencia en texto libre ("el juego se traba mucho", "la historia es aburrida") y el programa devuelve las **10 reseñas más parecidas**, ordenadas por similitud.

No es análisis de sentimiento: el rating se guarda para después, pero aquí solo importa **encontrar las opiniones que hablan de lo que se buscó**.

El "parecido" se mide con el **Modelo de Espacio Vectorial (VSM)**. Cada reseña es un vector con una dimensión por cada palabra (o par de palabras) del vocabulario. La consulta se convierte en un vector del mismo espacio y se compara con todas las reseñas usando la **similitud coseno**.

### Etapa actual

Esta entrega cubre la **recolección** (secciones 4 y 5) y la **normalización** (sección 6). Las representaciones vectoriales, la búsqueda y el benchmark (secciones 7-9) son la siguiente etapa.

| Entregable | Archivo | Estado |
|---|---|---|
| Código de recolección | `src/collect_steam.py`, `src/merge_corpus.py` | ✅ |
| Corpus crudo | `data/opinions_raw_corpus.tsv` (492) | ✅ |
| Versión cruda original | `data/raw/*/*.jsonl` (720) | ✅ |
| Distribución del corpus | `data/corpus_stats.txt` | ✅ |
| Código de normalización | `src/normalize.py`, `src/demo_normalize.py` | ✅ |
| Corpus normalizado | `data/opinions_normalized_corpus.tsv` | ✅ |

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

**Archivo:** `src/normalize.py` · **Comando:** `python main.py normalize` · **Demo:** `python main.py demo --id 416`

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

**Corrección hecha:** los verbos con pronombre pegado ("jugarlo", "comprarlo") son un solo token cuyo lema trae dos palabras ("jugar él"); antes eso dejaba un "el" suelto 266 veces en el corpus. Ahora se conserva solo el verbo. También se eliminan posesivos que spaCy etiqueta mal ("tus" como demostrativo).

**Efecto en el corpus:** cada reseña pasa de 95.8 a 56.4 palabras en promedio (41 % menos) y el vocabulario de 10,083 a 5,494 términos distintos (`python main.py demo --stats`).

---

## 4. Preguntas para repasar

1. ¿Por qué se usa POS para quitar stopwords en vez de una lista fija de palabras?
2. ¿Por qué no se quita "no"?
3. ¿Qué gana el modelo al lematizar ("jugadores" → "jugador")?
4. ¿Por qué se quitan los acentos?
5. ¿Por qué se piden por separado reseñas negativas y positivas?
6. ¿Por qué el `review_id` no puede cambiar una vez congelado el corpus?
