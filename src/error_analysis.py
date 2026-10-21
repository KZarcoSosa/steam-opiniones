"""Borrador verificable de los 5 mejores y 5 peores casos (sección 12).

Las hipótesis automáticas NO reemplazan la explicación humana; el equipo debe
corregirlas/confirmarlas antes de entregar el PDF final.
"""
import pandas as pd

try:
    from .vectorize import ROOT
    from .benchmark import RESULTS, QRELS, QUERIES
except ImportError:
    from vectorize import ROOT
    from benchmark import RESULTS, QRELS, QUERIES


def build_error_draft():
    m=pd.read_csv(RESULTS/'metrics_by_query.tsv',sep='\t',dtype={'query_id':str,'difficulty_level':str})
    rank=pd.read_csv(ROOT/'data/rankings.tsv',sep='\t',dtype={'query_id':str,'document_id':str})
    qrels=pd.read_csv(QRELS,sep='\t',dtype={'query_id':str,'document_id':str})
    q=pd.read_csv(QUERIES,sep='\t',dtype=str).set_index('query_id')
    norm=pd.read_csv(RESULTS/'normalized_queries.tsv',sep='\t',dtype=str).set_index('query_id')
    corpus=pd.read_csv(ROOT/'data/opinions_normalized_corpus.tsv',sep='\t',dtype=str).set_index('review_id')
    raw=pd.read_csv(ROOT/'data/opinions_raw_corpus.tsv',sep='\t',dtype=str).set_index('review_id')
    best=m.sort_values(['nDCG@10','P@10'],ascending=False).head(5)
    worst=m.sort_values(['nDCG@10','P@10']).head(5)
    rows=[]
    for tag,subset in [('buen_desempeño',best),('bajo_desempeño',worst)]:
        for _,row in subset.iterrows():
            qid=row['query_id']
            found=rank.loc[rank.query_id==qid,'document_id'].tolist()
            relevant=set(qrels.loc[qrels.query_id==qid,'document_id'])
            missed=sorted(relevant-set(found),key=int)
            unjudged=[doc for doc in found if doc not in relevant]
            selected=missed[0] if missed else ''
            terms=set(norm.loc[qid,'query_normalized'].split())
            tokens=set(corpus.loc[selected,'review_norm'].split()) if selected else set()
            overlap=len(terms & tokens)
            if not missed:
                cause='No hay relevantes omitidos entre los juicios anotados; revisar precisión y falsos positivos.'
            elif overlap <= 1:
                cause='Posible desajuste léxico o sinonimia: documento relevante omitido comparte pocos términos con la consulta.'
            else:
                cause='Posible competencia de términos frecuentes, longitud de reseñas o ponderación VSM; revisar posiciones.'
            rows.append({'query_id':qid,'group':tag,'query':q.loc[qid,'query'],
                         'P@10':row['P@10'],'nDCG@10':row['nDCG@10'],
                         'hits_top10':len(set(found)&relevant),'unjudged_top10':len(unjudged),
                         'unjudged_example':unjudged[0] if unjudged else '',
                         'relevant_omitted':len(missed),'omitted_example':selected,
                         'omitted_excerpt':' '.join(raw.loc[selected,'review'].split())[:220] if selected else '',
                         'query_document_common_terms':overlap,
                         'hypothesis_needs_human_validation':cause,
                         'team_analysis':''})
    out=pd.DataFrame(rows)
    out.to_csv(RESULTS/'error_analysis_draft.tsv',sep='\t',index=False,float_format='%.6f')
    return out
