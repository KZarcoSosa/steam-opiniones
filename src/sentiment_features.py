"""Etapa II, paso 2: preprocesamiento configurable y características lingüísticas.

Este módulo NO modifica nada de la Etapa I. Contiene transformadores compatibles con
scikit-learn (se usan dentro de Pipeline) para comparar, un factor a la vez:

  * TextPreprocessor  -> minúsculas/tokenización, lematización, palabras funcionales,
                          marcado de negación, repeticiones y corrección ortográfica.
  * LexiconFeatures   -> conteos de léxicos (SEL, LIWC), emojis/emoticonos y negadores.

Ambos son "sin estado" respecto al corpus (no aprenden nada de train), así que cachear su
resultado no introduce fuga de datos. Los vectorizadores y modelos SÍ se ajustan después,
solo con train, dentro del Pipeline.

Los textos de salida de TextPreprocessor van separados por espacios: el vectorizador que los
reciba debe usar token_pattern=r"\\S+" y lowercase=False para no perder emojis ni "NEG_xxx".

Demostración (no requiere modelos):
    python src/sentiment_features.py --text "No es nada bueno, sin embargo me encantó :)"
    python src/sentiment_features.py --id 14
    python src/sentiment_features.py --coverage      # cobertura de los léxicos sobre el corpus
"""
import argparse
import importlib.util
import pickle
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESOURCES = ROOT / "resources" / "lexicons"
SEL_FILE = RESOURCES / "SEL_full.txt"
LIWC_FILE = RESOURCES / "LIWC2007.dic"
EMOJI_FILE = RESOURCES / "Emojis_lexicon.xlsx"
LABELED = ROOT / "data" / "stage2" / "labeled_documents.tsv"
CACHE_DIR = ROOT / "data" / "stage2" / "cache"
SPACY_CACHE_FILE = CACHE_DIR / "spacy_analysis.pkl"
SPACY_MODEL = "es_core_news_sm"

# ----------------------------------------------------------------------------- expresiones
_BBCODE = re.compile(r"\[/?[a-zA-Z0-9*]+(=[^\]]*)?\]")           # igual que la Etapa I
_URL = re.compile(r"https?://\S+|www\.\S+")
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028\u2029\ufeff\ufe0f]")
_SPACES = re.compile(r"\s+")

_EMOTICON_SRC = (r"(?<!\w)(?:[:;=][\-o^']?[)(DdPpOo/\\|]|[)(][\-]?[:;=]|[Dd]:|[cC]:|:[cC]|<3|[xX][dD])(?!\w)")
_EMOJI_SRC = r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]"
EMOTICON_RE = re.compile(_EMOTICON_SRC)
EMOJI_RE = re.compile(_EMOJI_SRC)
EMO_SPLIT_RE = re.compile("(" + _EMOTICON_SRC + "|" + _EMOJI_SRC + ")")
TOKEN_RE = re.compile(_EMOTICON_SRC + "|" + _EMOJI_SRC +
                      r"|\d+(?:[.,/]\d+)*|[^\W\d_]+(?:['’][^\W\d_]+)?|[^\w\s]")
WORD_RE = re.compile(r"[^\W\d_]+")
_ELONGATION = re.compile(r"([^\W\d_])\1{2,}")                      # "buenooooo" -> "bueno"
_REPEATED_WORD = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.I)         # "muy muy bueno" -> "muy bueno"

# Negadores que NO deben tratarse como palabras vacías (el PDF lo prohíbe).
NEGATORS = {"no", "nunca", "jamas", "tampoco", "ni", "sin", "nada", "nadie",
            "ninguno", "ninguna", "ningun", "ningunos", "ningunas"}
# Palabras que cierran el alcance de la negación.
SCOPE_BREAKERS = {"pero", "aunque", "sino"}
# Palabras muy frecuentes que dentro del alcance no se marcan ni gastan la ventana
# (evita vocabulario basura como NEG_me, NEG_lo, NEG_es).
NEG_SKIP = {"el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "le", "les", "me", "te", "se", "nos",
            "a", "de", "en", "y", "e", "o", "u", "que", "es", "ser", "esta", "estar", "estan", "son", "haber",
            "hay", "ha", "he", "al", "del", "mi", "tu", "su"}
# Vocabulario del dominio que el corrector no debe "arreglar" (jerga, anglicismos, siglas).
DOMAIN_WORDS = {
    "fps", "bug", "bugs", "dlc", "dlcs", "mod", "mods", "lag", "lagea", "laguea", "lageo", "rpg", "jrpg",
    "pvp", "pve", "npc", "npcs", "boss", "bosses", "ping", "devs", "dev", "online", "offline", "gameplay",
    "parry", "rng", "tp", "ia", "ui", "hud", "steam", "skin", "skins", "loot", "nerf", "buff", "grind",
    "farmear", "farmeo", "spawn", "crash", "crashea", "crashear", "cheaters", "cheater", "hackers", "ranked",
    "matchmaking", "multijugador", "coop", "indie", "roguelike", "roguelite", "metroidvania", "pc", "gpu", "cpu",
    "ram", "ssd", "lol", "dota", "cs", "gg", "wp", "gl", "hf", "afk", "mmr", "smurf", "troll", "toxicos",
}

# Emoticonos frecuentes (el léxico de emojis solo cubre emojis). Puntaje (positivo, negativo).
EMOTICON_SCORES = {
    ":)": (1, 0), ":-)": (1, 0), "=)": (1, 0), ";)": (1, 0), ";-)": (1, 0), ":d": (1, 0), ":-d": (1, 0),
    "c:": (1, 0), "(:": (1, 0), "xd": (1, 0), "<3": (1, 0), ":p": (1, 0), ":-p": (1, 0),
    ":(": (0, 1), ":-(": (0, 1), "=(": (0, 1), ":'(": (0, 1), ":c": (0, 1), "d:": (0, 1), "):": (0, 1),
    ":/": (0, 1), ":-/": (0, 1), ":\\": (0, 1), ":|": (0, 0),
}


def strip_accents(word: str) -> str:
    """Quita tildes conservando la ñ (misma regla que la Etapa I)."""
    word = word.replace("ñ", "\0")
    word = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
    return word.replace("\0", "ñ")


def basic_clean(text: str) -> str:
    """Limpieza común a TODAS las configuraciones (BBCode, URLs, caracteres de control, espacios)."""
    text = _BBCODE.sub(" ", str(text))
    text = _URL.sub(" ", text)
    text = _CTRL.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _is_emo(token: str) -> bool:
    return bool(EMOTICON_RE.fullmatch(token) or EMOJI_RE.fullmatch(token))


def _kind(token: str) -> str:
    if _is_emo(token):
        return "emo"
    if re.fullmatch(r"[^\w\s]", token):
        return "punct"
    return "word"


def spacy_available() -> bool:
    try:
        return importlib.util.find_spec("spacy") is not None and \
            importlib.util.find_spec(SPACY_MODEL) is not None
    except Exception:
        return False


# ----------------------------------------------------------------------------- spaCy (con caché)
_SPACY_CACHE = {}
_SPACY_CACHE_LOADED = False


def _load_spacy_cache():
    global _SPACY_CACHE_LOADED
    if _SPACY_CACHE_LOADED:
        return
    _SPACY_CACHE_LOADED = True
    if SPACY_CACHE_FILE.is_file():
        try:
            with open(SPACY_CACHE_FILE, "rb") as fh:
                _SPACY_CACHE.update(pickle.load(fh))
        except Exception:
            pass                                    # caché corrupta: se reconstruye sola


def _save_spacy_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPACY_CACHE_FILE, "wb") as fh:
        pickle.dump(_SPACY_CACHE, fh)


def _spacy_analyze(segments):
    """Analiza segmentos de texto (sin emojis) con el MISMO modelo y reglas de la Etapa I.
    Devuelve, por segmento, tuplas (superficie_minuscula, tipo, lema_sin_tildes, conservar_etapa1)."""
    _load_spacy_cache()
    missing = [s for s in dict.fromkeys(segments) if s not in _SPACY_CACHE]
    if missing:
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        import normalize                              # Etapa I: importa spaCy solo si se necesita
        nlp = normalize.get_nlp()
        for seg, doc in zip(missing, nlp.pipe(missing, batch_size=64)):
            items = []
            for tok in doc:
                if tok.is_space:
                    continue
                if tok.is_punct or tok.pos_ in ("PUNCT", "SYM", "SPACE"):
                    items.append((tok.text, "punct", tok.text, False))
                    continue
                lemma = normalize.token_lemma(tok) if tok.lemma_.strip() else strip_accents(tok.lower_)
                items.append((tok.lower_, "word", lemma, bool(normalize.keep_token(tok))))
            _SPACY_CACHE[seg] = items
        _save_spacy_cache()
    return [_SPACY_CACHE[s] for s in segments]


# ----------------------------------------------------------------------------- preprocesador
_OUT_CACHE = {}          # (parámetros) -> {texto: salida}; no depende de train


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Convierte reseñas crudas en texto normalizado separado por espacios.

    Parámetros (todos controlables para experimentos de un solo factor):
      lemmatize            lemas de spaCy sin tildes (igual que la Etapa I). Requiere spaCy.
      remove_function_words  quita artículos, preposiciones, conjunciones y pronombres (regla de la Etapa I).
      protect_negators     si True, no/nunca/jamás/tampoco/ni/sin/nada/nadie se conservan aunque sean
                           palabras funcionales. (False reproduce la Etapa I, que sí los elimina.)
      negation             marca el alcance de la negación: tras un negador, las siguientes
                           `neg_window` palabras de contenido reciben el prefijo NEG_ (hasta
                           puntuación, emoji, pero/aunque/sino; artículos, clíticos y "es/ser/estar"
                           no se marcan ni gastan la ventana).
                           "sin embargo" se trata como conector, no como negación.
      repeats              letras alargadas (3+) -> 1 y palabras consecutivas repetidas -> 1.
      spelling             corrección ortográfica con pyspellchecker (es, distancia 1, con caché por
                           palabra en data/stage2/cache/); no toca nombres propios,
                           siglas, palabras de dominio ni palabras con dígitos.
      protected_words      conjunto extra de palabras que el corrector no debe tocar.
    Sin ninguna opción activa, solo limpia, tokeniza y pasa a minúsculas (línea base mínima).
    """

    def __init__(self, lemmatize=False, remove_function_words=False, protect_negators=True,
                 negation=False, repeats=False, spelling=False, neg_window=3, protected_words=None):
        self.lemmatize = lemmatize
        self.remove_function_words = remove_function_words
        self.protect_negators = protect_negators
        self.negation = negation
        self.repeats = repeats
        self.spelling = spelling
        self.neg_window = neg_window
        self.protected_words = protected_words

    # sklearn: no aprende nada
    def fit(self, X, y=None):
        return self

    def _protected(self):
        extra = set(self.protected_words) if self.protected_words else set()
        return DOMAIN_WORDS | {w.lower() for w in extra}

    def _key(self):
        return (self.lemmatize, self.remove_function_words, self.protect_negators, self.negation,
                self.repeats, self.spelling, self.neg_window, tuple(sorted(self._protected())) if self.spelling else ())

    # --- pasos a nivel de cadena -------------------------------------------------------------
    def _pre_text(self, text, speller):
        text = basic_clean(text)
        if self.repeats:
            text = _ELONGATION.sub(r"\1", text)
            text = _REPEATED_WORD.sub(r"\1", text)
        if self.spelling:
            text = self._correct(text, speller)
        return text

    def _correct(self, text, speller):
        protected = self._protected()

        def fix(m):
            w = m.group(0)
            low = w.lower()
            before = text[:m.start()].rstrip()
            sentence_start = (before == "" or before[-1] in ".!?¡¿:\n")
            if (len(w) <= 3 or low in protected or (w[0].isupper() and not sentence_start)
                    or (w.isupper() and len(w) > 1) or low in speller):
                return w
            fixed = _correct_word(speller, low)
            return fixed if fixed else w

        return WORD_RE.sub(fix, text)

    # --- tokenización -------------------------------------------------------------------------
    def _items_regex(self, text):
        return [(t.lower(), _kind(t), t.lower(), True) for t in TOKEN_RE.findall(text)]

    def _items_spacy(self, texts):
        """Separa emojis/emoticonos (spaCy los parte mal) y analiza el resto con spaCy."""
        parts_per_text, segments = [], []
        for text in texts:
            parts = [p.strip() for p in EMO_SPLIT_RE.split(text) if p and p.strip()]
            parts_per_text.append(parts)
            segments.extend(p for p in parts if not _is_emo(p))
        analyzed = dict(zip(segments, _spacy_analyze(segments))) if segments else {}
        out = []
        for parts in parts_per_text:
            items = []
            for p in parts:
                if _is_emo(p):
                    items.append((p.lower(), "emo", p.lower(), True))
                else:
                    items.extend(analyzed[p])
            out.append(items)
        return out

    # --- ensamblado final ---------------------------------------------------------------------
    def _build(self, items):
        """items: (superficie, tipo, lema, conservar_etapa1) -> texto final."""
        use_spacy_forms = self.lemmatize or self.remove_function_words
        n = len(items)
        out, remaining = [], 0
        for i, (surface, kind, lemma, keep1) in enumerate(items):
            if kind == "punct":
                remaining = 0
                continue
            if kind == "emo":
                out.append(surface)
                remaining = 0                      # un emoji/emoticono cierra la cláusula
                continue
            plain = strip_accents(surface)
            nxt = strip_accents(items[i + 1][0]) if i + 1 < n else ""
            is_neg = plain in NEGATORS and not (plain == "sin" and nxt == "embargo")
            if plain in SCOPE_BREAKERS or (plain == "sin" and nxt == "embargo"):
                remaining = 0
            form = lemma if self.lemmatize else surface
            keep = True
            if self.remove_function_words:
                keep = keep1 or (self.protect_negators and is_neg)
            if is_neg:
                remaining = self.neg_window if self.negation else 0
                if keep:
                    out.append(form)
                continue
            if not keep:
                continue
            if self.negation and remaining > 0:
                if plain in NEG_SKIP or strip_accents(lemma) in NEG_SKIP:
                    out.append(form)               # función frecuente: ni se marca ni cuenta
                else:
                    out.append("NEG_" + form)
                    remaining -= 1
            else:
                out.append(form)
        return " ".join(out)

    def transform(self, X):
        texts = [("" if x is None else str(x)) for x in X]
        cache = _OUT_CACHE.setdefault(self._key(), {})
        todo = [t for t in dict.fromkeys(texts) if t not in cache]
        if todo:
            speller = _get_speller() if self.spelling else None
            pre = [self._pre_text(t, speller) for t in todo]
            if self.lemmatize or self.remove_function_words:
                items_list = self._items_spacy(pre)
            else:
                items_list = [self._items_regex(p) for p in pre]
            for t, items in zip(todo, items_list):
                cache[t] = self._build(items)
            if self.spelling:
                _save_spell_cache()
        return [cache[t] for t in texts]


_SPELLER = None
SPELL_CACHE_FILE = CACHE_DIR / "spelling_es.pkl"
_SPELL_WORDS = {}          # palabra (minúsculas) -> corrección ("" = sin cambio)
_SPELL_LOADED = False
_SPELL_DIRTY = False
MAX_SPELL_LEN = 18         # palabras más largas se dejan igual (y se evita el costo)


def _get_speller():
    """Corrector en español con distancia de edición 1: la distancia 2 (valor por defecto del paquete)
    genera cientos de miles de candidatos por palabra y tarda minutos; además corrige de más."""
    global _SPELLER
    if _SPELLER is None:
        try:
            from spellchecker import SpellChecker
        except ImportError:
            sys.exit("Falta pyspellchecker. Instálalo con:  pip install pyspellchecker")
        print("Ortografía: usando pyspellchecker (es, distancia 1). La primera vez tarda; luego queda en caché.",
              flush=True)
        _SPELLER = SpellChecker(language="es", distance=1)
    return _SPELLER


def _load_spell_cache():
    global _SPELL_LOADED
    if _SPELL_LOADED:
        return
    _SPELL_LOADED = True
    if SPELL_CACHE_FILE.is_file():
        try:
            with open(SPELL_CACHE_FILE, "rb") as fh:
                _SPELL_WORDS.update(pickle.load(fh))
        except Exception:
            pass


def _save_spell_cache():
    global _SPELL_DIRTY
    if _SPELL_DIRTY:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(SPELL_CACHE_FILE, "wb") as fh:
            pickle.dump(_SPELL_WORDS, fh)
        _SPELL_DIRTY = False


def _correct_word(speller, low):
    """Corrección de una palabra ya en minúsculas, con caché (cada palabra distinta se calcula una sola vez)."""
    global _SPELL_DIRTY
    _load_spell_cache()
    if low not in _SPELL_WORDS:
        fixed = speller.correction(low) if len(low) <= MAX_SPELL_LEN else None
        _SPELL_WORDS[low] = fixed if (fixed and fixed != low) else ""
        _SPELL_DIRTY = True
    return _SPELL_WORDS[low]


# ----------------------------------------------------------------------------- léxicos
_RES_CACHE = {}


def _need(path: Path):
    if not path.is_file():
        sys.exit(f"Falta el recurso {path}\nCopia el archivo ahí (ver resources/lexicons/README.md).")


def load_sel():
    """SEL (Spanish Emotion Lexicon): {palabra_sin_tildes: [(emoción, PFA), ...]}."""
    if "sel" not in _RES_CACHE:
        _need(SEL_FILE)
        sel = {}
        for line in SEL_FILE.read_text(encoding="utf-8").splitlines()[1:]:
            p = line.split("\t")
            if len(p) < 7 or not p[0].strip():
                continue
            sel.setdefault(strip_accents(p[0].strip().lower()), []).append((p[6].strip(), float(p[5])))
        _RES_CACHE["sel"] = sel
    return _RES_CACHE["sel"]


# Mismo criterio que el script de ejemplo del curso: positivo = alegría + sorpresa;
# negativo = enojo + miedo + repulsión + tristeza.
SEL_POSITIVE = {"Alegría", "Sorpresa"}
SEL_NEGATIVE = {"Enojo", "Miedo", "Repulsión", "Tristeza"}


def load_liwc():
    """LIWC 2007 en español: (conjunto_exactas, prefijos) para EmoPos y EmoNeg."""
    if "liwc" not in _RES_CACHE:
        _need(LIWC_FILE)
        raw = LIWC_FILE.read_text(encoding="utf-8").split("%")
        cats = {}
        for line in raw[1].strip().splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2:
                cats[parts[0]] = parts[1]              # id -> nombre de categoría
        pos_id = next(i for i, n in cats.items() if n == "EmoPos")
        neg_id = next(i for i, n in cats.items() if n == "EmoNeg")
        res = {"pos": (set(), set()), "neg": (set(), set())}
        for line in raw[2].strip().splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            word, ids = parts[0].lower(), set(parts[1:])
            for key, cid in (("pos", pos_id), ("neg", neg_id)):
                if cid in ids:
                    (res[key][1] if word.endswith("*") else res[key][0]).add(word.rstrip("*"))
        _RES_CACHE["liwc"] = res
    return _RES_CACHE["liwc"]


def _liwc_hit(token, lex):
    exact, prefixes = lex
    if token in exact:
        return True
    return any(token[:i] in prefixes for i in range(len(token), 2, -1))


def load_emoji():
    """Léxico de emojis: {emoji: (positivo, negativo)}. Procedencia/licencia: pendiente de confirmar con el profesor."""
    if "emoji" not in _RES_CACHE:
        _need(EMOJI_FILE)
        df = pd.read_excel(EMOJI_FILE)
        d = {}
        for _, r in df.iterrows():
            e = str(r["emoji"]).replace("\ufe0f", "").strip()
            d[e] = (float(r["positive"]), float(r["negative"]))
        _RES_CACHE["emoji"] = d
    return _RES_CACHE["emoji"]


ALL_BLOCKS = ("sel", "liwc", "emoji", "negation")


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """Características numéricas NO negativas (sirven también para MultinomialNB) a partir del texto crudo.

    blocks: subconjunto de ('sel','liwc','emoji','negation'); permite hacer ablaciones.
      sel       sel_pos, sel_neg      suma de PFA de las palabras SEL (lematizadas, sin tildes)
      liwc      liwc_pos, liwc_neg    conteo de palabras LIWC EmoPos / EmoNeg (formas sin lematizar)
      emoji     emoji_pos, emoji_neg  suma de puntajes de emojis (léxico) y emoticonos (tabla propia)
      negation  n_negators            conteo de negadores (excluye "sin embargo")
    lemma_fn: función opcional texto -> lista de lemas (para pruebas). Por defecto usa spaCy con caché.
    """

    def __init__(self, blocks=ALL_BLOCKS, lemma_fn=None):
        self.blocks = blocks
        self.lemma_fn = lemma_fn

    def fit(self, X, y=None):
        return self

    @property
    def feature_names_(self):
        names = []
        if "sel" in self.blocks:
            names += ["sel_pos", "sel_neg"]
        if "liwc" in self.blocks:
            names += ["liwc_pos", "liwc_neg"]
        if "emoji" in self.blocks:
            names += ["emoji_pos", "emoji_neg"]
        if "negation" in self.blocks:
            names += ["n_negators"]
        return names

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)

    def _lemmas(self, texts):
        if self.lemma_fn is not None:
            return [self.lemma_fn(t) for t in texts]
        pre = TextPreprocessor(lemmatize=True, remove_function_words=False, negation=False)
        return [s.split() for s in pre.transform(texts)]

    def transform(self, X):
        texts = [("" if x is None else str(x)) for x in X]
        blocks = set(self.blocks)
        unknown = blocks - set(ALL_BLOCKS)
        if unknown:
            raise ValueError(f"Bloques desconocidos: {unknown}")
        sel = load_sel() if "sel" in blocks else None
        liwc = load_liwc() if "liwc" in blocks else None
        emoji = load_emoji() if "emoji" in blocks else None
        lemmas = self._lemmas(texts) if "sel" in blocks else None
        rows = []
        for i, text in enumerate(texts):
            cleaned = basic_clean(text)
            tokens = [t.lower() for t in TOKEN_RE.findall(cleaned)]
            row = []
            if sel is not None:
                pos = neg = 0.0
                for lemma in lemmas[i]:
                    for emotion, w in sel.get(strip_accents(lemma), ()):
                        if emotion in SEL_POSITIVE:
                            pos += w
                        elif emotion in SEL_NEGATIVE:
                            neg += w
                row += [pos, neg]
            if liwc is not None:
                words = [t for t in tokens if WORD_RE.fullmatch(t)]
                row += [sum(_liwc_hit(w, liwc["pos"]) for w in words),
                        sum(_liwc_hit(w, liwc["neg"]) for w in words)]
            if emoji is not None:
                ep = en = 0.0
                for t in tokens:
                    if EMOJI_RE.fullmatch(t):
                        p, n = emoji.get(t, (0.0, 0.0))
                        ep, en = ep + p, en + n
                    elif EMOTICON_RE.fullmatch(t):
                        p, n = EMOTICON_SCORES.get(t, (0, 0))
                        ep, en = ep + p, en + n
                row += [ep, en]
            if "negation" in blocks:
                plain = [strip_accents(t) for t in tokens]
                row.append(sum(1 for j, t in enumerate(plain)
                               if t in NEGATORS and not (t == "sin" and j + 1 < len(plain) and plain[j + 1] == "embargo")))
            rows.append(row)
        return np.asarray(rows, dtype=float).reshape(len(texts), len(self.feature_names_))


# ----------------------------------------------------------------------------- presets y demo
PRESETS = {
    "P0_minimo": dict(),
    "P1_etapa1": dict(lemmatize=True, remove_function_words=True, protect_negators=False),
    "P2_lemas": dict(lemmatize=True),
    "P3_negacion": dict(negation=True),
    "P4_lemas_negacion": dict(lemmatize=True, negation=True),
    "P5_negacion_repeticiones": dict(negation=True, repeats=True),
    "P6_negacion_ortografia": dict(negation=True, spelling=True),
}
NEEDS_SPACY = {"P1_etapa1", "P2_lemas", "P4_lemas_negacion"}


def _demo(text):
    print(f"\nTEXTO: {text}\n")
    have_spacy = spacy_available()
    for name, params in PRESETS.items():
        if name in NEEDS_SPACY and not have_spacy:
            print(f"{name:26s} (omitido: spaCy/{SPACY_MODEL} no disponible)")
            continue
        if params.get("spelling"):
            try:
                import spellchecker  # noqa: F401
            except ImportError:
                print(f"{name:26s} (omitido: falta pyspellchecker)")
                continue
        print(f"{name:26s} {TextPreprocessor(**params).transform([text])[0]}")
    if all(p.exists() for p in (SEL_FILE, LIWC_FILE, EMOJI_FILE)):
        fallback = None if have_spacy else (lambda t: [strip_accents(w.lower()) for w in WORD_RE.findall(t)])
        lf = LexiconFeatures(lemma_fn=fallback)
        vals = lf.transform([text])[0]
        print("\nLexiconFeatures:", dict(zip(lf.feature_names_, np.round(vals, 3))),
              "" if have_spacy else "(SEL sin lematizar: spaCy no disponible)")
    else:
        print("\n(Faltan archivos en resources/lexicons/ para mostrar LexiconFeatures)")


def _coverage():
    df = pd.read_csv(LABELED, sep="\t", dtype=str, keep_default_na=False)
    have_spacy = spacy_available()
    fallback = None if have_spacy else (lambda t: [strip_accents(w.lower()) for w in WORD_RE.findall(t)])
    lf = LexiconFeatures(lemma_fn=fallback)
    mat = lf.transform(df["text"].tolist())
    print(f"Documentos: {len(df)} | SEL {'con lemas (spaCy)' if have_spacy else 'SIN lematizar (spaCy no disponible)'}")
    for j, name in enumerate(lf.feature_names_):
        print(f"  {name:12s} documentos con valor > 0: {(mat[:, j] > 0).sum():4d} ({(mat[:, j] > 0).mean():.0%})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text")
    ap.add_argument("--id", help="document_id de labeled_documents.tsv")
    ap.add_argument("--coverage", action="store_true")
    a = ap.parse_args()
    if a.coverage:
        _coverage()
    elif a.id:
        df = pd.read_csv(LABELED, sep="\t", dtype=str, keep_default_na=False).set_index("document_id")
        _demo(df.loc[a.id, "text"])
    elif a.text:
        _demo(a.text)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()