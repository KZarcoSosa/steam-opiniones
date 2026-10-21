"""Sección 9. Benchmark de 15 consultas y 5 métricas de recuperación.

  python main.py benchmark --validate
  python main.py benchmark
  python main.py benchmark --all

Los qrels son juicios independientes de las predicciones del sistema; deben ser
revisados por el equipo antes de reportarlos como verdad-terreno definitiva.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from .normalize import normalize_text
    from .vectorize import ROOT, CONTENTS, FEATURES, WEIGHTS, checksum
    from .retrieve import load_model, rank_normalized_query
except ImportError:
    from normalize import normalize_text
    from vectorize import ROOT, CONTENTS, FEATURES, WEIGHTS, checksum
    from retrieve import load_model, rank_normalized_query

BENCHMARK = ROOT / 'data' / 'benchmark'
QUERIES = BENCHMARK / 'queries.tsv'
QRELS = BENCHMARK / 'qrels.tsv'
LOCK = BENCHMARK / 'benchmark_lock.json'
RESULTS = ROOT / 'data' / 'results'


def load_benchmark(require_lock=True):
    q = pd.read_csv(QUERIES, sep='\t', dtype=str, keep_default_na=False)
    r = pd.read_csv(QRELS, sep='\t', dtype=str, keep_default_na=False)
    if q.columns.tolist() != ['query_id', 'difficulty_level', 'query']:
        raise ValueError('Formato esperado queries.tsv: query_id, difficulty_level, query')
    if r.columns.tolist() != ['query_id', 'document_id', 'relevance']:
        raise ValueError('Formato esperado qrels.tsv: query_id, document_id, relevance')
    if len(q) != 15 or q.query_id.duplicated().any() or q['query'].eq('').any():
        raise ValueError('Se necesitan 15 consultas no vacías con IDs únicos')
    if set(q.query_id) != {f'Q{i:02d}' for i in range(1, 16)}:
        raise ValueError('IDs de consultas esperados: Q01..Q15')
    if Counter(q.difficulty_level) != {'1': 5, '2': 5, '3': 5}:
        raise ValueError('Se requieren 5 consultas fáciles, 5 medias, 5 difíciles (1/2/3)')
    if r.duplicated(['query_id', 'document_id']).any():
        raise ValueError('Se repite una pareja consulta/documento en qrels.tsv')
    if not set(r.relevance).issubset({'1', '2'}) or not set(r.query_id).issubset(set(q.query_id)):
        raise ValueError('qrels.tsv sólo debe contener relevancias 1/2, con consultas conocidas')
    corpus = pd.read_csv(ROOT / 'data' / 'opinions_raw_corpus.tsv', sep='\t', dtype=str, keep_default_na=False)
    if not set(r.document_id).issubset(set(corpus.review_id)):
        raise ValueError('qrels contiene documentos que no existen en el corpus')
    for qid, block in r.groupby('query_id'):
        if not 10 <= len(block) <= 15 or set(block.relevance) != {'1', '2'}:
            raise ValueError(f'{qid}: se requieren 10–15 documentos, con relevancias 1 y 2')
    if set(r.query_id) != set(q.query_id):
        raise ValueError('Faltan evaluaciones de relevancia de algunas consultas')
    if require_lock:
        if not LOCK.exists():
            raise ValueError('Falta benchmark_lock.json; revisar y congelar el benchmark antes de evaluar')
        obj = json.loads(LOCK.read_text(encoding='utf-8'))
        for name, p in [('queries_sha256', QUERIES), ('qrels_sha256', QRELS)]:
            if obj.get(name) != checksum(p):
                raise ValueError(f'El benchmark cambió ({p.name}); revisar y volver a congelar')
        if obj.get('corpus_sha256') != checksum(ROOT / 'data' / 'opinions_raw_corpus.tsv'):
            raise ValueError('Corpus crudo diferente del usado para anotar relevancias')
    return q, r


def metrics_for_query(ranking, judgments, k=10):
    """ranking: [(document_id, cosine_score), ...] de TODO el corpus.

    P/R/F1/MRR usan rel>0. nDCG usa ganancias graduales (2**rel-1).
    RR se obtiene de la primera coincidencia en el ranking completo, no sólo Top-10.
    """
    judgments = {str(doc): int(rel) for doc, rel in judgments.items()}
    top = ranking[:k]
    hits = sum(judgments.get(str(doc), 0) > 0 for doc, _ in top)
    p = hits / k
    r = hits / len(judgments) if judgments else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    reciprocal_rank = next((1 / i for i, (doc, _) in enumerate(ranking, start=1)
                            if judgments.get(str(doc), 0) > 0), 0.0)
    dcg = sum((2 ** judgments.get(str(doc), 0) - 1) / np.log2(i + 1)
              for i, (doc, _) in enumerate(top, start=1))
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(i + 1) for i, rel in enumerate(ideal, 1))
    return {'P@10': p, 'R@10': r, 'F1@10': f1, 'RR': reciprocal_rank,
            'nDCG@10': dcg / idcg if idcg else 0.0, 'relevant_in_top10': hits,
            'relevant_total': len(judgments)}


def normalize_queries(queries):
    # Sólo se normaliza aquí una vez y la MISMA función se usa en búsqueda.
    return {row.query_id: normalize_text(row.query) for row in queries.itertuples(index=False)}


def evaluate_one(q, r, normalized, content, features, weight):
    corpus, vectorizer, matrix = load_model(content, features, weight)
    judgments = {qid: dict(zip(block.document_id, block.relevance)) for qid, block in r.groupby('query_id')}
    ranking_rows, metric_rows = [], []
    for row in q.itertuples(index=False):
        ranking = rank_normalized_query(normalized[row.query_id], corpus, vectorizer, matrix)
        for pos, (doc_id, score) in enumerate(ranking[:10], 1):
            ranking_rows.append({'query_id': row.query_id, 'document_id': doc_id,
                                 'rank': pos, 'score': round(score, 8)})
        metric_rows.append({'query_id': row.query_id, 'difficulty_level': row.difficulty_level,
                            **metrics_for_query(ranking, judgments[row.query_id])})
    return pd.DataFrame(ranking_rows), pd.DataFrame(metric_rows)


def make_summary(metrics):
    rows = []
    for level, block in [('1', metrics[metrics.difficulty_level == '1']),
                         ('2', metrics[metrics.difficulty_level == '2']),
                         ('3', metrics[metrics.difficulty_level == '3']),
                         ('overall', metrics)]:
        rows.append({'difficulty_level': level, 'queries': len(block),
                     **{name: block[name].mean() for name in ['P@10', 'R@10', 'F1@10', 'RR', 'nDCG@10']}})
    return pd.DataFrame(rows).rename(columns={'RR': 'MRR'})


def main():
    parser = argparse.ArgumentParser(description='Validar y evaluar 15 consultas con qrels congelados')
    parser.add_argument('--validate', action='store_true', help='Sólo validar archivos sin necesitar spaCy')
    parser.add_argument('--approve', action='store_true', help='Tras revisar los 180 juicios, validar la aprobación del equipo y recongelar el benchmark')
    parser.add_argument('--all', action='store_true', help='Comparar las 12 configuraciones además de guardar la predeterminada')
    parser.add_argument('--content', choices=list(CONTENTS), default='review')
    parser.add_argument('--features', choices=list(FEATURES), default='unigram')
    parser.add_argument('--weight', choices=list(WEIGHTS), default='tfidf')
    args = parser.parse_args()
    if args.approve:
        q, r = load_benchmark(require_lock=False)
        audit = pd.read_csv(BENCHMARK / 'annotation_audit.tsv', sep='\t', dtype=str, keep_default_na=False)
        if len(audit) != len(r) or not audit.human_checked.str.lower().eq('yes').all():
            raise ValueError('Cada una de las 180 filas de annotation_audit.tsv debe marcar human_checked=yes tras revisión real')
        merged = r.merge(audit, on=['query_id','document_id','relevance'], how='outer', indicator=True)
        if not merged['_merge'].eq('both').all():
            raise ValueError('Las etiquetas del archivo de auditoría y qrels.tsv no coinciden; sincronízalas.')
        lock = {'version': 'validated-v1', 'judgment_status': 'checked_by_team',
                'queries_sha256': checksum(QUERIES), 'qrels_sha256': checksum(QRELS),
                'corpus_sha256': checksum(ROOT / 'data' / 'opinions_raw_corpus.tsv'),
                'approved_at_utc': datetime.now(timezone.utc).isoformat()}
        LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Benchmark verificado y congelado tras revisión del equipo.')
        return
    q, r = load_benchmark()
    print(f'Benchmark verificado: {len(q)} consultas y {len(r)} juicios positivos.')
    if args.validate:
        print('Integridad de consultas, qrels e IDs: OK. Aún requiere revisión humana de relevancias.')
        return
    normalized = normalize_queries(q)
    if any(not text for text in normalized.values()):
        raise ValueError('Una consulta quedó vacía al normalizar con spaCy')
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'query_id': k, 'query_normalized': v} for k, v in normalized.items()]).to_csv(
        RESULTS / 'normalized_queries.tsv', sep='\t', index=False)
    review_status = json.loads(LOCK.read_text(encoding='utf-8')).get('judgment_status')
    if review_status != 'checked_by_team':
        print('AVISO: qrels en borrador no validados por el equipo; resultados sólo exploratorios.')
    ranks, metrics = evaluate_one(q, r, normalized, args.content, args.features, args.weight)
    summary = make_summary(metrics)
    ranks.to_csv(ROOT / 'data' / 'rankings.tsv', sep='\t', index=False)
    metrics.to_csv(RESULTS / 'metrics_by_query.tsv', sep='\t', index=False, float_format='%.6f')
    summary.to_csv(RESULTS / 'metrics_summary.tsv', sep='\t', index=False, float_format='%.6f')
    metadata = {'content': args.content, 'features': args.features, 'weight': args.weight,
                'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
                'corpus_sha256': checksum(ROOT / 'data' / 'opinions_raw_corpus.tsv'),
                'queries_sha256': checksum(QUERIES), 'qrels_sha256': checksum(QRELS),
                'normalizer': 'spaCy es_core_news_sm; src.normalize.normalize_text',
                'judgment_status': review_status}
    (RESULTS / 'run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        from .error_analysis import build_error_draft
    except ImportError:
        from error_analysis import build_error_draft
    draft = build_error_draft()
    print(f'Borrador de análisis de errores: {len(draft)} casos (5 buenos, 5 malos).')
    print('\nMétricas de la configuración predeterminada/elegida:')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nGuardados rankings y métricas en {RESULTS.parent}')
    if args.all:
        rows = []
        for content in CONTENTS:
            for features in FEATURES:
                for weight in WEIGHTS:
                    _, table = evaluate_one(q, r, normalized, content, features, weight)
                    sm = make_summary(table).iloc[-1]
                    rows.append({'content': content, 'features': features, 'weight': weight,
                                 **{key: sm[key] for key in ['P@10', 'R@10', 'F1@10', 'MRR', 'nDCG@10']}})
        compare = pd.DataFrame(rows).sort_values('nDCG@10', ascending=False)
        compare.to_csv(RESULTS / 'all_configurations.tsv', sep='\t', index=False, float_format='%.6f')
        print('\nComparación de 12 configuraciones por nDCG@10 (diagnóstica):')
        print(compare.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
