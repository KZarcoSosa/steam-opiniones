"""Sección 7: 12 representaciones VSM (2 campos x 2 n-gramas x 3 pesos).

Uso: python main.py vectorize
No requiere repetir la normalización: usa la versión congelada del corpus.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / 'data' / 'opinions_normalized_corpus.tsv'
MODELS = ROOT / 'models' / 'v1'
CONTENTS = {'review': 'review_norm', 'item_review': 'item_review_norm'}
FEATURES = {'unigram': (1, 1), 'bigram': (2, 2)}
WEIGHTS = ('frequency', 'binary', 'tfidf')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_name(content, features, weight):
    if content not in CONTENTS or features not in FEATURES or weight not in WEIGHTS:
        raise ValueError('Configuración inválida de VSM')
    return f'{content}_{features}_{weight}'


def load_corpus():
    if not CORPUS.is_file():
        raise FileNotFoundError(f'No existe {CORPUS}. Primero ejecuta: python main.py normalize')
    df = pd.read_csv(CORPUS, sep='\t', dtype=str, keep_default_na=False)
    if not {'review_id', *CONTENTS.values()}.issubset(df.columns):
        raise ValueError('Faltan columnas del corpus normalizado')
    if df.review_id.duplicated().any() or df.review_id.eq('').any():
        raise ValueError('review_id duplicado o vacío en corpus normalizado')
    return df


def build(force=False):
    df = load_corpus()
    MODELS.mkdir(parents=True, exist_ok=True)
    manifest = {
        'version': 'corpus-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'corpus_sha256': checksum(CORPUS),
        'corpus_count': len(df),
        'document_ids': df.review_id.tolist(),
        'normalizer': 'src.normalize.normalize_text (spaCy es_core_news_sm)',
        'configurations': [],
    }
    for content, column in CONTENTS.items():
        for features, ngram_range in FEATURES.items():
            for weight in WEIGHTS:
                name = artifact_name(content, features, weight)
                if weight == 'tfidf':
                    vec = TfidfVectorizer(ngram_range=ngram_range, lowercase=False,
                                          token_pattern=r'(?u)\b\w+\b', norm='l2')
                else:
                    vec = CountVectorizer(ngram_range=ngram_range, lowercase=False,
                                          token_pattern=r'(?u)\b\w+\b', binary=weight == 'binary')
                matrix = vec.fit_transform(df[column].tolist())
                joblib.dump(vec, MODELS / f'{name}.joblib', compress=3)
                save_npz(MODELS / f'{name}.npz', matrix, compressed=True)
                manifest['configurations'].append({
                    'content': content, 'features': features, 'weight': weight,
                    'name': name, 'matrix_shape': list(matrix.shape),
                    'nonzeros': int(matrix.nnz),
                })
                print(f'{name:33s} docs={matrix.shape[0]:3d} features={matrix.shape[1]:6d}')
    (MODELS / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n12 configuraciones guardadas en: {MODELS}')
    return manifest


if __name__ == '__main__':
    build()
