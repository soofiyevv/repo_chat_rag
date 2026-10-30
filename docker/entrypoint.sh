#!/bin/sh
set -e

MODEL="${EMBED_MODEL:-bge-m3}"
MARKER="$DB_PATH/.indexed-$MODEL"

case "$1" in
  api)
    if [ ! -s "$CHUNKS_FILE" ]; then
      echo "Chunking $REPO_URL"
      python chunker.py "$REPO_URL" -o "$CHUNKS_FILE"
    fi
    if [ ! -f "$MARKER" ]; then
      echo "Building index with $MODEL"
      python index.py --model "$MODEL"
      touch "$MARKER"
    fi
    exec uvicorn api:app --host 0.0.0.0 --port 8000
    ;;
  reindex)
    rm -f "$CHUNKS_FILE" "$DB_PATH"/.indexed-*
    python chunker.py "$REPO_URL" -o "$CHUNKS_FILE"
    python index.py --model "$MODEL"
    touch "$MARKER"
    ;;
  bot)
    exec python bot.py
    ;;
  *)
    exec "$@"
    ;;
esac
