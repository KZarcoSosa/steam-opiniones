"""
Recolector de reseñas de Steam (API pública de reseñas, sin login ni API key).

Uso:
    python src/collect_steam.py --member kevin
    python src/collect_steam.py --member kevin --per-game 80 --languages spanish latam
    python src/collect_steam.py --member kevin --overwrite      # rehace tus archivos desde cero

Lee los juegos asignados al integrante en config/apps.json y guarda, por juego,
un archivo JSONL "crudo" en data/raw/<member>/<appid>.jsonl (texto original sin tocar).
Esos JSONL son lo que se sube al repo; el TSV final lo arma merge_corpus.py.

Endpoints usados (públicos, documentados por Valve):
  - https://store.steampowered.com/appreviews/<appid>?json=1 ...
  - https://store.steampowered.com/api/appdetails?appids=<appid>
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "apps.json"
RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"

REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
DETAILS_URL = "https://store.steampowered.com/api/appdetails"
HEADERS = {"User-Agent": "PLN-Practica2-OpinionRetrieval/1.0 (proyecto escolar)"}
SLEEP_S = 1.5          # pausa entre peticiones para respetar límites
MAX_RETRIES = 4
DAY_RANGE = 365       # ventana máxima documentada por Steam para filter=all
MIN_WORDS = 15         # reseñas más cortas ("10/10", "buen juego") no sirven para recuperación


def get_json(url, params, log):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 30 * attempt
                log.warning("429 (rate limit) en %s, esperando %ss", url, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            log.error("Intento %d/%d falló para %s %s: %s", attempt, MAX_RETRIES, url, params, e)
            time.sleep(5 * attempt)
    return None


def get_app_details(appid, log):
    data = get_json(DETAILS_URL, {"appids": appid, "l": "spanish"}, log)
    time.sleep(SLEEP_S)
    try:
        d = data[str(appid)]["data"]
        genres = [g["description"] for g in d.get("genres", [])]
        return d.get("name", str(appid)), genres
    except (TypeError, KeyError):
        log.error("No se pudieron obtener detalles de app %s", appid)
        return str(appid), []


def collect_app(appid, languages, per_game, out_path, log, review_filter="all", negative_share=0.5):
    """Baja reseñas negativas y positivas por separado para no quedar con un corpus
    95 % positivo. review_filter='all' = ordenadas por utilidad (abarcan varios años);
    'recent' = las más nuevas."""
    name, genres = get_app_details(appid, log)
    seen = set()
    counts = {"negative": 0, "positive": 0}
    if out_path.exists():  # permite re-ejecutar sin duplicar
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                seen.add(rec["recommendationid"])
                counts["positive" if rec.get("voted_up") else "negative"] += 1
    log.info("[%s] %s | géneros=%s | ya existentes=%s", appid, name, genres, counts)

    targets = {"negative": round(per_game * negative_share)}
    targets["positive"] = per_game - targets["negative"]

    with open(out_path, "a", encoding="utf-8") as out:
        for review_type in ("negative", "positive"):
            for lang in languages:
                _collect(appid, name, genres, lang, review_type, review_filter,
                             targets[review_type], counts, seen, out, log)
                if counts[review_type] >= targets[review_type]:
                    break
        # Si un juego casi no tiene negativas, completar con positivas (y viceversa)
        for review_type in ("positive", "negative"):
            for lang in languages:
                if sum(counts.values()) >= per_game:
                    break
                _collect(appid, name, genres, lang, review_type, review_filter,
                         counts[review_type] + per_game - sum(counts.values()),
                         counts, seen, out, log)
    log.info("[%s] final: %s", appid, counts)
    return sum(counts.values())


def _collect(appid, name, genres, lang, review_type, review_filter, target, counts, seen, out, log):
    """Pagina la API para un idioma y tipo (negative/positive) hasta llegar a target."""
    cursor, pages = "*", 0
    while counts[review_type] < target:
        params = {
            "json": 1, "filter": review_filter, "language": lang,
            "review_type": review_type, "purchase_type": "all",
            "num_per_page": 100, "cursor": cursor,
        }
        if review_filter == "all":
            # Sin esto Steam solo busca en los últimos ~30 días (de ahí que todo saliera de 2026)
            params["day_range"] = DAY_RANGE
        data = get_json(REVIEWS_URL.format(appid=appid), params, log)
        time.sleep(SLEEP_S)
        if not data or data.get("success") != 1:
            log.error("[%s] respuesta inválida (lang=%s, tipo=%s, cursor=%s)", appid, lang, review_type, cursor)
            break
        reviews = data.get("reviews", [])
        if not reviews:
            break
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for rv in reviews:
            rid = rv["recommendationid"]
            text = rv.get("review", "")
            if rid in seen or len(text.split()) < MIN_WORDS:
                continue
            seen.add(rid)
            record = {
                "recommendationid": rid,
                "appid": appid,
                "item_name": name,
                "genres": genres,
                "language": rv.get("language", lang),
                "review": text,                      # texto original sin modificar
                "voted_up": rv.get("voted_up"),
                "votes_up": rv.get("votes_up"),
                "votes_funny": rv.get("votes_funny"),
                "playtime_at_review_min": rv.get("author", {}).get("playtime_at_review"),
                "timestamp_created": rv.get("timestamp_created"),
                "timestamp_updated": rv.get("timestamp_updated"),
                "steam_purchase": rv.get("steam_purchase"),
                "received_for_free": rv.get("received_for_free"),
                "retrieved_at": retrieved_at,
                # No guardamos steamid del autor: no hace falta y evita datos personales.
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["positive" if record["voted_up"] else "negative"] += 1
            if counts[review_type] >= target:
                break
        pages += 1
        new_cursor = data.get("cursor")
        if not new_cursor or new_cursor == cursor:
            break
        cursor = new_cursor
    log.info("[%s] lang=%s tipo=%s páginas=%d -> %s", appid, lang, review_type, pages, counts)


def main():
    ap = argparse.ArgumentParser(description="Recolecta reseñas de Steam para un integrante.")
    ap.add_argument("--member", required=True, help="Nombre del integrante en config/apps.json")
    ap.add_argument("--per-game", type=int, default=None, help="Reseñas a conservar por juego")
    ap.add_argument("--filter", default=None, choices=["all", "recent", "updated"],
                    help="all = más útiles (varios años, por defecto); recent = más nuevas")
    ap.add_argument("--negative-share", type=float, default=None,
                    help="Proporción de reseñas negativas por juego (por defecto 0.5)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Borra lo que ya tengas de tus juegos y vuelve a bajar desde cero")
    ap.add_argument("--languages", nargs="+", default=None,
                    help="Idiomas de Steam (spanish = España, latam = Latinoamérica, english)")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.member not in cfg["members"]:
        raise SystemExit(f"'{args.member}' no está en config/apps.json. Opciones: {list(cfg['members'])}")
    per_game = args.per_game or cfg.get("per_game", 60)
    languages = args.languages or cfg.get("languages", ["latam", "spanish"])
    review_filter = args.filter or cfg.get("filter", "all")
    negative_share = args.negative_share if args.negative_share is not None else cfg.get("negative_share", 0.5)

    LOG_DIR.mkdir(exist_ok=True)
    out_dir = RAW_DIR / args.member
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / f"collect_{args.member}.log", encoding="utf-8"),
                  logging.StreamHandler()],
    )
    log = logging.getLogger("collect")

    total = 0
    for appid in cfg["members"][args.member]:
        out_path = out_dir / f"{appid}.jsonl"
        if args.overwrite and out_path.exists():
            out_path.write_text("", encoding="utf-8")
        total += collect_app(appid, languages, per_game, out_path, log, review_filter, negative_share)
    log.info("Listo. %d reseñas en %s", total, out_dir)


if __name__ == "__main__":
    main()
