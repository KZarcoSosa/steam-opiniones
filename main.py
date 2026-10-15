"""
Interfaz única del proyecto (sección 10: comandos claramente separados).

    python main.py collect --member kevin      # 1. Recolección / actualización del corpus
    python main.py merge                       #    Unir lo de los 3 -> opinions_raw_corpus.tsv
    python main.py build                       # 2. Normalización + representaciones vectoriales
    python main.py search "consulta" [--content review|item_review] [--features unigram|bigram] [--rep frequency|binary|tfidf]
                                               # 3. Consulta -> Top-10
    python main.py benchmark [--all]           # 4. Benchmark y métricas

Cada comando solo llama al script correspondiente en src/, que también se
puede ejecutar por separado.
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
COMMANDS = {
    "collect": ["collect_steam.py"],
    "merge": ["merge_corpus.py"],
    "build": ["normalize.py", "vectorize.py"],
    "search": ["search.py"],
    "benchmark": ["benchmark.py"],
    "pool": ["label_pool.py"],
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    cmd, extra = sys.argv[1], sys.argv[2:]
    for script in COMMANDS[cmd]:
        args = extra if len(COMMANDS[cmd]) == 1 else []
        code = subprocess.call([sys.executable, str(SRC / script), *args])
        if code:
            sys.exit(code)


if __name__ == "__main__":
    main()
