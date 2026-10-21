"""Sección 11: informe PDF reproducible a partir de la colección y de las métricas.

Cuando todavía no hay evaluación auténtica, se crea un reporte de avance que
NO inventa métricas ni describe casos exitosos/fracasados no verificados.
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, KeepTogether

try:
    from .vectorize import ROOT
    from .benchmark import BENCHMARK, RESULTS
except ImportError:
    from vectorize import ROOT
    from benchmark import BENCHMARK, RESULTS

OUT = ROOT / 'docs' / 'reporte_etapa1.pdf'


def fmt(x):
    return f'{float(x):.3f}'


def main():
    reg='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    if Path(reg).is_file() and Path(bold).is_file():
        pdfmetrics.registerFont(TTFont('DejaVu',reg))
        pdfmetrics.registerFont(TTFont('DejaVu-Bold',bold))
        font, fontb = 'DejaVu','DejaVu-Bold'
    else:
        font, fontb = 'Helvetica', 'Helvetica-Bold'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TituloPLN',fontName=fontb,fontSize=16,leading=21,spaceAfter=13,textColor=colors.HexColor('#214268')))
    styles.add(ParagraphStyle(name='SeccionPLN',fontName=fontb,fontSize=11,leading=15,spaceBefore=14,spaceAfter=6,textColor=colors.HexColor('#214268')))
    styles.add(ParagraphStyle(name='TextoPLN',fontName=font,fontSize=8.7,leading=13,spaceAfter=7))
    styles.add(ParagraphStyle(name='MiniPLN',fontName=font,fontSize=7.2,leading=10))
    styles.add(ParagraphStyle(name='HeaderPLN',fontName=fontb,fontSize=7.3,leading=10,textColor=colors.white))
    doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=1.65*cm,leftMargin=1.65*cm,topMargin=1.8*cm,bottomMargin=1.65*cm)
    story=[]
    add=lambda t: story.append(Paragraph(t,styles['TextoPLN']))
    section=lambda t:story.append(Paragraph(t,styles['SeccionPLN']))
    def tab(headers, data, widths=None):
        rows=[[Paragraph(str(x),styles['HeaderPLN']) for x in headers]]
        rows += [[Paragraph(str(x).replace('&','&amp;').replace('<','&lt;'),styles['MiniPLN']) for x in row] for row in data]
        tbl=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2C4D75')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F4F7')]),
            ('GRID',(0,0),(-1,-1),0.22,colors.HexColor('#C7D1DB')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ]))
        story.append(tbl)
    raw=pd.read_csv(ROOT/'data/opinions_raw_corpus.tsv',sep='\t',dtype=str,keep_default_na=False)
    norm=pd.read_csv(ROOT/'data/opinions_normalized_corpus.tsv',sep='\t',dtype=str,keep_default_na=False)
    queries=pd.read_csv(BENCHMARK/'queries.tsv',sep='\t',dtype=str,keep_default_na=False)
    qrels=pd.read_csv(BENCHMARK/'qrels.tsv',sep='\t',dtype=str,keep_default_na=False)
    status=json.loads((BENCHMARK/'benchmark_lock.json').read_text(encoding='utf-8')).get('judgment_status')
    metapath=RESULTS/'run_metadata.json'
    evaluated=(RESULTS/'metrics_by_query.tsv').is_file() and metapath.is_file()
    metadata=json.loads(metapath.read_text(encoding='utf-8')) if evaluated else {}
    story.append(Paragraph('Recuperación de opiniones de Steam — Etapa 1',styles['TituloPLN']))
    add('<b>Asignatura:</b> Procesamiento de Lenguaje Natural. <b>Equipo:</b> Kevin, Ángel y Erick.')
    add(f'<b>Estado:</b> {"evaluación ejecutada" if evaluated else "reporte de avance; evaluación pendiente"}. '
        f'<b>Juicios de relevancia:</b> {"revisados por el equipo" if status=="checked_by_team" else "borrador asistido, requieren aprobación humana"}.')
    section('1. Objetivo y corpus')
    add('El sistema busca opiniones que discuten un aspecto o experiencia expresados como consulta libre. '
        'Utiliza representación en espacio vectorial y similitud coseno; el rating binario de Steam se conserva sólo como metadato.')
    tab(['Indicador','Resultado'],[
        ('Documentos crudos y normalizados',f'{len(raw)} / {len(norm)}'),
        ('Videojuegos',str(raw.item_name.nunique())),
        ('Fuente','API pública de reseñas de Steam'),
        ('Valoraciones','; '.join(f'{k}: {v}' for k,v in sorted(raw.rating.value_counts().items()))),
        ('Período reseñas',f'{raw.date.min()} a {raw.date.max()}'),
        ('Muestreo','12 videojuegos x 41 reseñas; idiomas latam y spanish'),
    ],[8*cm,9.7*cm])
    section('2. Normalización y representaciones VSM')
    add('Se parte de los textos normalizados mediante spaCy (POS, eliminación selectiva de artículos, preposiciones, '
        'conjunciones y pronombres, lematización y limpieza). Para consultas nuevas, el programa exige exactamente '
        'el mismo normalizador; no aplica un proceso alternativo silencioso.')
    add('Se crearon 12 configuraciones: corpus [reseña / nombre del juego + reseña] × características '
        '[unigrama / bigrama] × ponderación [frecuencia / binaria / TF-IDF]. Cada configuración '
        'conserva vectorizador y matriz dispersa para el cálculo de coseno.')
    section('3. Consultas y juicios de relevancia')
    add(f'El benchmark contiene {len(queries)} consultas (5 fáciles, 5 medias y 5 difíciles) y '
        f'{len(qrels)} juicios iniciales (12 por consulta, relevancias 1 o 2). '
        'Los IDs se corresponden con review_id del corpus congelado. No se incluyen relevancias 0 en qrels.')
    tab(['ID','Nivel','Consulta'],[(r.query_id,r.difficulty_level,r.query) for r in queries.itertuples(index=False)],[1.5*cm,1.5*cm,14.7*cm])
    section('4. Evaluación cuantitativa')
    if evaluated:
        per=pd.read_csv(RESULTS/'metrics_by_query.tsv',sep='\t')
        summary=pd.read_csv(RESULTS/'metrics_summary.tsv',sep='\t')
        add(f'<b>Configuración:</b> {metadata.get("content")}, {metadata.get("features")}, {metadata.get("weight")}. '
            'P@10, R@10 y F1@10 consideran relevantes grados 1 y 2; MRR utiliza el primer relevante '
            'del ranking completo; nDCG@10 utiliza grados 1 y 2 con ganancias exponenciales.')
        rows=[]
        for _,r in summary.iterrows():
            rows.append([r.difficulty_level,*[fmt(r[name]) for name in ['P@10','R@10','F1@10','MRR','nDCG@10']]])
        tab(['Nivel','P@10','R@10','F1@10','MRR','nDCG@10'],rows,[2.9*cm]*6)
        section('5. Análisis de casos: mejores y peores consultas')
        top=per.sort_values(['nDCG@10','P@10'],ascending=False).head(5)
        low=per.sort_values(['nDCG@10','P@10']).head(5)
        def explain_group(name,df):
            add('<b>'+name+'</b>')
            tab(['Consulta','P@10','nDCG@10'],
                [[r.query_id,fmt(r['P@10']),fmt(r['nDCG@10'])] for _,r in df.iterrows()],
                [3.2*cm,3*cm,4*cm])
        explain_group('Cinco con mejor desempeño',top)
        explain_group('Cinco con menor desempeño',low)
        draft_path=RESULTS/'error_analysis_draft.tsv'
        if draft_path.is_file():
            diag=pd.read_csv(draft_path,sep='\t',dtype=str,keep_default_na=False)
            for rr in diag.itertuples(index=False):
                add(f'<b>{rr.query_id}:</b> relevantes omitidos {rr.relevant_omitted}; '
                    f'resultados sin relevancia positiva anotada en top-10: {rr.unjudged_top10}. '
                    f'Hipótesis: {rr.hypothesis_needs_human_validation}')
        add('<b>Guía de interpretación:</b> examinar falsos positivos y relevantes omitidos por '
            'sinónimos, escaso solapamiento léxico, vocabulario específico, consultas ambiguas '
            'y fallas del etiquetado POS/lematización. El archivo de auditoría debe completarse '
            'con observaciones específicas antes de entregar el análisis de errores como final.')
    else:
        add('<b>PENDIENTE, NO SE INVENTAN RESULTADOS:</b> se han construido las matrices, '
            'pero aún falta ejecutar consultas con el modelo spaCy es_core_news_sm. '
            'Tras instalarlo y validar las anotaciones: <b>python main.py benchmark --all</b>. '
            'El sistema generará rankings.tsv, métricas por consulta, medias y una comparativa de 12 configuraciones.')
        section('5. Análisis de errores: protocolo previo a la ejecución')
        add('Seleccionar las cinco consultas de mayor desempeño y las cinco de menor desempeño usando '
            'nDCG@10 y P@10. Examinar reseñas devueltas/no devueltas y documentar causas '
            '(sinonimia, ambigüedad, vocabulario, desequilibrio léxico, normalización). '
            'No es posible identificar honradamente los mejores/peores casos sin rankings evaluados.')
    section('6. Reproducibilidad y limitaciones')
    add('Flujo: <b>python main.py vectorize</b> → revisar annotation_audit.tsv y qrels.tsv '
        '→ <b>python main.py benchmark --approve</b> → <b>python main.py benchmark --all</b> '
        '→ <b>python main.py report</b>. La colección original se conserva: no ejecutar merge '
        'después de congelar el benchmark sin volver a revisar los identificadores.')
    add('Limitaciones: las consultas difíciles pueden tener pocos términos compartidos con las reseñas; '
        'bigramas exactos suelen dar vectores de consulta vacíos y los lemas de jerga pueden ser ruidosos. '
        'El rendimiento reportado sobre los juicios usados para elegir la configuración no es una estimación '
        'independiente de generalización.')
    def page(canvas,doc):
        canvas.saveState();canvas.setFont(font,7)
        canvas.setFillColor(colors.HexColor('#667788'))
        canvas.drawString(1.65*cm,1.0*cm,'PLN / Recuperación de opiniones / Etapa 1')
        canvas.drawRightString(A4[0]-1.65*cm,1.0*cm,str(doc.page))
        canvas.restoreState()
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print('Reporte creado:', OUT)

if __name__ == '__main__':
    main()
