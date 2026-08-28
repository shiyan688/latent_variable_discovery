#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bridge_root="$project_root/runs/starry_zt_scale_aware_neural_bridge_20260829"
python_bin="$project_root/.venv-lvs-gpu/bin/python"
runner="$project_root/scripts/run_starry_zt_scale_aware_neural_bridge_20260829.py"

mkdir -p "$bridge_root/logs" "$bridge_root/runtime_cache"

run_worker() {
    local worker=$1
    local index fold seed log_root cache_root
    for ((index=worker; index<15; index+=4)); do
        fold=$((index / 3))
        seed=$((index % 3))
        log_root="$bridge_root/logs/fold${fold}_seed${seed}.log"
        cache_root="$bridge_root/runtime_cache/worker${worker}"
        MPLCONFIGDIR="$cache_root/mpl" \
        XDG_CACHE_HOME="$cache_root/xdg" \
        PYTHONPYCACHEPREFIX="$cache_root/pycache" \
        "$python_bin" "$runner" \
            --fold "$fold" \
            --seed "$seed" \
            --threads 4 \
            >"$log_root" 2>&1
    done
}

worker_pids=()
for worker in 0 1 2 3; do
    run_worker "$worker" &
    worker_pids+=("$!")
done

for worker_pid in "${worker_pids[@]}"; do
    wait "$worker_pid"
done
