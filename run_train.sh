#!/usr/bin/env bash
# Convenience launcher. Safe to re-run: training resumes automatically.
set -euo pipefail
cd "$(dirname "$0")"
python tools/verify_setup.py || { echo "setup check failed"; exit 1; }
exec python train.py \
  --run-name ppo_btc_eth_ltc \
  --epochs "${EPOCHS:-2000}" \
  --episodes-per-epoch "${EPISODES:-8}" \
  --decision-interval "${INTERVAL:-5}" \
  "$@"
