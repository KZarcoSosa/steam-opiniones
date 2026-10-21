"""Pruebas sin dependencia del modelo spaCy descargable."""
import sys
from pathlib import Path
import math
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.vectorize import load_corpus, ROOT, CONTENTS, FEATURES, WEIGHTS
from src.retrieve import load_model, rank_normalized_query
from src.benchmark import load_benchmark, metrics_for_query


def test_corpus_and_qrels_integrity():
    corpus=load_corpus()
    q,r=load_benchmark()
    assert len(corpus)==492
    assert len(q)==15 and len(r)==180
    assert set(r.relevance)=={'1','2'}


@pytest.mark.parametrize('content',list(CONTENTS))
@pytest.mark.parametrize('features',list(FEATURES))
@pytest.mark.parametrize('weight',list(WEIGHTS))
def test_all_saved_models_and_ranks(content,features,weight):
    df, vec, matrix=load_model(content,features,weight)
    assert matrix.shape[0]==len(df)==492
    assert len(vec.get_feature_names_out())==matrix.shape[1]
    token='juego' if features=='unigram' else 'juego mundo'
    ranks=rank_normalized_query(token,df,vec,matrix)
    assert len(ranks)==492
    assert len({id for id,score in ranks})==492
    assert all(ranks[i][1]>=ranks[i+1][1] for i in range(491))
    assert all(-0.0000001<=score<=1.0000001 for _,score in ranks)


def test_metrics_match_professor_pdf_example():
    names=['N003','N050','N012','N018','N060','N024','N070','N031','N080','N090']
    ranking=[(ident,float(10-i)) for i,ident in enumerate(names)]
    qrels={'N003':2,'N007':1,'N012':2,'N018':1,'N024':2,'N031':1}
    metrics=metrics_for_query(ranking,qrels)
    assert metrics['P@10']==0.5
    assert metrics['R@10']==pytest.approx(5/6)
    assert metrics['F1@10']==pytest.approx(0.625)
    assert metrics['RR']==1
    assert metrics['nDCG@10']==pytest.approx(0.835,abs=0.001)


def test_zero_query_gives_no_positive_cosines():
    df,vec,matrix=load_model()
    result=rank_normalized_query('terminoinexistenteabsoluto',df,vec,matrix)
    assert all(score==0 for _,score in result)
    assert result[0][0]=='1'
