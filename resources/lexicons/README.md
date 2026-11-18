# Recursos léxicos de la Etapa II

Copia aquí (con estos nombres exactos) los archivos proporcionados en la materia:

| Archivo | Recurso | Idioma | Uso en el proyecto |
|---|---|---|---|
| `SEL_full.txt` | Spanish Emotion Lexicon (6 emociones con peso PFA) | Español | `sel_pos` (alegría + sorpresa) y `sel_neg` (enojo + miedo + repulsión + tristeza), sobre lemas sin tildes |
| `LIWC2007.dic` | LIWC 2007, categorías EmoPos y EmoNeg | Español | `liwc_pos`, `liwc_neg`: conteo sobre formas sin lematizar (admite comodines `palabra*`) |
| `Emojis_lexicon.xlsx` | Puntajes de emociones, positivo y negativo por emoji (359 emojis) | Independiente del idioma | `emoji_pos`, `emoji_neg`; emoticonos con una tabla propia pequeña |

## Descartados
- `SentiWordNet_3.0.0.txt` y `opinion-lexicon-English/`: son léxicos en **inglés**; las reseñas del corpus están en español.

## Licencias y procedencia (completar antes de entregar)
- LIWC 2007 es un recurso con licencia comercial: **no lo publiques en un repositorio público**. Si el repo es público, agrega `resources/lexicons/LIWC2007.dic` al `.gitignore`.
- Procedencia y licencia del léxico de emojis y de SEL: pendientes de confirmar con el profesor.

## Limitaciones a discutir en el reporte
- Desajuste de dominio: palabras como "matar", "enemigo" o "difícil" no expresan necesariamente emociones negativas en reseñas de videojuegos.
- Los puntajes del léxico de emojis están sesgados hacia lo positivo (también para emojis como 💀), y solo ~5 % de las reseñas contiene emojis.
- SEL asigna "sorpresa" al polo positivo (criterio del script de ejemplo del curso), lo cual es discutible.
