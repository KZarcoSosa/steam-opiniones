"""Sección 8: consulta de texto libre, normalización idéntica al corpus y top-10.

Ejemplo:
python main.py search --query "se me cae el rendimiento y hay tirones" --content review \\
  --features unigram --weight tfidf
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .normalize import normalize_text
    from .vectorize import ROOT, CORPUS, MODELS, artifact_name, checksum, load_corpus
except ImportError:
    from normalize import normalize_text
    from vectorize import ROOT, CORPUS, MODELS, artifact_name, checksum, load_corpus

RAW = ROOT / 'data' / 'opinions_raw_corpus.tsv'


def load_model(content='review', features='unigram', weight='tfidf'):
    name = artifact_name(content, features, weight)
    manifest_path = MODELS / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('No hay modelos. Ejecuta primero: python main.py vectorize')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['corpus_sha256'] != checksum(CORPUS):
        raise RuntimeError('Corpus diferente al modelado; ejecuta python main.py vectorize y valida los qrels.')
    df = load_corpus()
    if df.review_id.tolist() != manifest['document_ids']:
        raise RuntimeError('Desalineación entre identificadores del corpus y los vectores.')
    vectorizer = joblib.load(MODELS / f'{name}.joblib')
    matrix = load_npz(MODELS / f'{name}.npz')
    if matrix.shape[0] != len(df):
        raise RuntimeError('Dimensiones del modelo inconsistentes con el corpus.')
    return df, vectorizer, matrix


def rank_normalized_query(normalized_query, df, vectorizer, matrix):
    """Núcleo recuperador. La llamada pública debe normalizar ANTES con spaCy.

    Devuelve TODO el ranking (inclusive ceros, usando review_id como desempate),
    para calcular MRR global, aunque la interfaz sólo presente 10 resultados.
    """
    qvec = vectorizer.transform([normalized_query])
    sims = cosine_similarity(qvec, matrix).ravel()
    # lexsort usa la última columna como llave principal; empate por ID numérico.
    ids = df.review_id.astype(int).to_numpy()
    order = np.lexsort((ids, -sims))
    return [(str(df.review_id.iloc[i]), float(sims[i])) for i in order]


def search(query, content='review', features='unigram', weight='tfidf', top=10):
    # EXACTAMENTE la misma función utilizada para producir el corpus normalizado.
    qnorm = normalize_text(query)
    if not qnorm:
        raise ValueError('La consulta no contiene términos tras la normalización.')
    df, vectorizer, matrix = load_model(content, features, weight)
    ranking = rank_normalized_query(qnorm, df, vectorizer, matrix)
    raw = pd.read_csv(RAW, sep='\t', dtype=str, keep_default_na=False).set_index('review_id')
    results = []
    for position, (doc_id, score) in enumerate(ranking[:top], 1):
        if doc_id not in raw.index:
            raise RuntimeError(f'No existe review_id={doc_id} en el corpus crudo')
        row = raw.loc[doc_id]
        results.append({'rank': position, 'document_id': doc_id, 'score': score,
                        'item_name': row.item_name, 'rating': row.rating,
                        'review': row.review, 'url': row.url})
    return qnorm, results


def main():
    parser = argparse.ArgumentParser(description='Busca las 10 opiniones más similares con VSM + coseno')
    parser.add_argument('--query', required=True, help='Consulta libre en español')
    parser.add_argument('--content', choices=['review', 'item_review'], default='review')
    parser.add_argument('--features', choices=['unigram', 'bigram'], default='unigram')
    parser.add_argument('--weight', choices=['frequency', 'binary', 'tfidf'], default='tfidf')
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    if args.top <= 0:
        parser.error('--top debe ser positivo')
    qnorm, results = search(args.query, args.content, args.features, args.weight, args.top)
    print(f'Consulta normalizada: {qnorm}')
    print(f'Comparación: {args.content} | {args.features} | {args.weight}')
    for item in results:
        short = ' '.join(item['review'].split())[:300]
        print(f"\n{item['rank']:2d}. doc={item['document_id']}  coseno={item['score']:.5f} "
              f"| {item['item_name']} | rating={item['rating']}\n    {short}")


if __name__ == '__main__':
    main()
