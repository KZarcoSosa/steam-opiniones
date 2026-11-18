"""Sección 8: consulta de texto libre, normalización idéntica al corpus y top-10.

Ejemplo:
python main.py search --query "se me cae el rendimiento y hay tirones" --content review \\
  --features unigram --weight tfidf

Por nivel del benchmark (corre las 5 consultas de ese nivel y marca las relevantes):
python main.py search --nivel 1 --features bigram --weight tfidf
python main.py search --qid Q02 --features bigram --weight tfidf
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
QUERIES = ROOT / 'data' / 'benchmark' / 'queries.tsv'
QRELS = ROOT / 'data' / 'benchmark' / 'qrels.tsv'
LEVEL = {'1': 'fácil', '2': 'media', '3': 'difícil'}


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


def print_results(qnorm, results, args, rels=None):
    print(f'Consulta normalizada: {qnorm}')
    print(f'Comparación: {args.content} | {args.features} | {args.weight}')
    for item in results:
        short = ' '.join(item['review'].split())[:300]
        mark = ''
        if rels is not None:
            r = rels.get(item['document_id'], 0)
            mark = f'  [relevancia={r}]' if r else '  [no relevante]'
        print(f"\n{item['rank']:2d}. doc={item['document_id']}  coseno={item['score']:.5f} "
              f"| {item['item_name']} | rating={item['rating']}{mark}\n    {short}")


def run_benchmark_queries(args):
    q = pd.read_csv(QUERIES, sep='\t', dtype=str)
    qrels = pd.read_csv(QRELS, sep='\t', dtype=str)
    q = q[q.query_id == args.qid.upper()] if args.qid else q[q.difficulty_level == args.nivel]
    if q.empty:
        raise SystemExit('No hay consultas con ese nivel o ID.')
    summary = []
    for row in q.itertuples():
        rels = {d: int(r) for d, r in qrels[qrels.query_id == row.query_id]
                [['document_id', 'relevance']].itertuples(index=False) if int(r) > 0}
        print('\n' + '#' * 100)
        print(f'{row.query_id} | nivel {row.difficulty_level} ({LEVEL[row.difficulty_level]}) | {row.query}')
        print('#' * 100)
        try:
            qnorm, results = search(row.query, args.content, args.features, args.weight, args.top)
        except ValueError as e:
            print(e); summary.append((row.query_id, row.query, 0, len(rels))); continue
        print_results(qnorm, results, args, rels)
        hits = sum(1 for it in results if it['document_id'] in rels)
        print(f'\n  -> {hits} de {len(results)} resultados son relevantes '
              f'(P@{args.top} = {hits / args.top:.2f}); el benchmark tiene {len(rels)} relevantes.')
        summary.append((row.query_id, row.query, hits, len(rels)))
    if len(summary) > 1:
        print('\n' + '=' * 100)
        print(f'RESUMEN | {args.content} | {args.features} | {args.weight}')
        print('=' * 100)
        for qid, text, hits, n in sorted(summary, key=lambda x: -x[2]):
            print(f'{qid}  P@{args.top} {hits / args.top:.2f}  ({hits}/{args.top}, {n} relevantes)  {text}')
        print(f'Promedio P@{args.top}: {sum(h for _, _, h, _ in summary) / (args.top * len(summary)):.3f}')


def main():
    parser = argparse.ArgumentParser(description='Busca las 10 opiniones más similares con VSM + coseno')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--query', help='Consulta libre en español')
    group.add_argument('--nivel', choices=['1', '2', '3'], help='Corre las consultas del benchmark de ese nivel')
    group.add_argument('--qid', help='Corre una consulta del benchmark por su ID (p. ej. Q02)')
    parser.add_argument('--content', choices=['review', 'item_review'], default='review')
    parser.add_argument('--features', choices=['unigram', 'bigram'], default='unigram')
    parser.add_argument('--weight', choices=['frequency', 'binary', 'tfidf'], default='tfidf')
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    if args.top <= 0:
        parser.error('--top debe ser positivo')
    if args.nivel or args.qid:
        return run_benchmark_queries(args)
    qnorm, results = search(args.query, args.content, args.features, args.weight, args.top)
    print_results(qnorm, results, args)


if __name__ == '__main__':
    main()
