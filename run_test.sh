#!/usr/bin/env bash
# Avvia la versione di test della demo sulla porta 8502.
# Uso:  ./run_test.sh            (dalla cartella del progetto)
#       PORT=8600 ./run_test.sh  (per usare un'altra porta)
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8502}"
export DEMO_LABEL="${DEMO_LABEL:-TEST}"
exec streamlit run app.py --server.port "$PORT"
