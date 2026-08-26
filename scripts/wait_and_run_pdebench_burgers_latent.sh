#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
runner="${repo_root}/scripts/run_pdebench_burgers_latent_study.py"
data_path="${repo_root}/data/external/pdebench/1D_Burgers_Sols_Nu0.02.hdf5"
output_root="${repo_root}/runs/pdebench_burgers_latent_20260809"

mkdir -p "${output_root}"
while [[ ! -f "${data_path}" ]]; do
  partial_size=0
  if [[ -f "${data_path}.part" ]]; then
    partial_size="$(stat -c %s "${data_path}.part")"
  fi
  printf '%s waiting_for_verified_data partial_bytes=%s\n' "$(date --iso-8601=seconds)" "${partial_size}"
  sleep 60
done

"${python_bin}" "${runner}" prepare \
  --data-path "${data_path}" \
  --output-root "${output_root}" \
  --data-seed 20260809 \
  --train-labels 64 \
  --validation-labels 16 \
  --test-labels 32 \
  --x-points 32 \
  --t-points 16

while true; do
  gpu_state="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  if printf '%s\n' "${gpu_state}" | awk -F, '
    {
      index=$1 + 0
      memory=$2 + 0
      utilization=$3 + 0
      if (index >= 4 && index <= 7) {
        count += 1
        if (memory < 2000 && utilization < 20) ready += 1
      }
    }
    END { exit !(count == 4 && ready == 4) }
  '; then
    printf '%s gpu_4_7_ready\n' "$(date --iso-8601=seconds)"
    break
  fi
  compact_state="$(printf '%s\n' "${gpu_state}" | awk -F, '$1 + 0 >= 4 && $1 + 0 <= 7 {printf "gpu%s:%sMiB/%s%% ", $1 + 0, $2 + 0, $3 + 0}')"
  printf '%s waiting_for_gpu_4_7 %s\n' "$(date --iso-8601=seconds)" "${compact_state}"
  sleep 60
done

"${python_bin}" "${runner}" launch \
  --q-dims 4,8,16 \
  --methods joint_mse,alternating_mse \
  --seeds 0,1,2 \
  --gpus 4,5,6,7 \
  --output-root "${output_root}" \
  --epochs 300 \
  --support-ratio 0.3 \
  --batch-size 256 \
  --resume \
  --save-artifacts

"${python_bin}" "${repo_root}/scripts/analyze_pdebench_burgers_latent.py" \
  --output-root "${output_root}" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260809
