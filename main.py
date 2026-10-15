"""
Interfaz del proyecto (sección 10: comandos claramente separados).

    python main.py collect --member kevin      # 1. Recolección / actualización del corpus
    python main.py merge                       #    Unir lo de los 3 -> opinions_raw_corpus.tsv
    python main.py normalize                   # 2. Normalización -> opinions_normalized_corpus.tsv
    python main.py demo [--id 416 | --text "..." | --stats]
                                               #    Demostración paso a paso de la normalización

Cada comando solo llama al script correspondiente en src/, que también se
puede ejecutar por separado.
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
COMMANDS = {
    "collect": "collect_steam.py",
    "merge": "merge_corpus.py",
    "normalize": "normalize.py",
    "demo": "demo_normalize.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    script, extra = COMMANDS[sys.argv[1]], sys.argv[2:]
    sys.exit(subprocess.call([sys.executable, str(SRC / script), *extra]))


if __name__ == "__main__":
    main()
