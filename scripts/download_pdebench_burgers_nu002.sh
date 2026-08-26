#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${repo_root}/data/external/pdebench"
target="${data_dir}/1D_Burgers_Sols_Nu0.02.hdf5"
partial="${target}.part"
expected_size="8232968312"
expected_sha256="b1d1ef10a612abef7eedd99873323289416a53c737c6cf04cb59c90020ed1911"
expected_md5="7c8c717a3a7818145877baa57106b090"
url="https://huggingface.co/datasets/pdebench/Burgers/resolve/main/1D_Burgers_Sols_Nu0.02.hdf5?download=true"

mkdir -p "${data_dir}"

if [[ -f "${target}" ]]; then
  actual_size="$(stat -c %s "${target}")"
  if [[ "${actual_size}" == "${expected_size}" ]] && printf '%s  %s\n' "${expected_sha256}" "${target}" | sha256sum --check --status; then
    printf 'already_valid %s\n' "${target}"
    exit 0
  fi
  printf 'existing target failed validation: %s bytes\n' "${actual_size}" >&2
  exit 1
fi

curl --location --fail --retry 8 --retry-delay 5 --continue-at - \
  --output "${partial}" "${url}"

actual_size="$(stat -c %s "${partial}")"
if [[ "${actual_size}" != "${expected_size}" ]]; then
  printf 'size mismatch: expected=%s actual=%s\n' "${expected_size}" "${actual_size}" >&2
  exit 1
fi

printf '%s  %s\n' "${expected_sha256}" "${partial}" | sha256sum --check
printf '%s  %s\n' "${expected_md5}" "${partial}" | md5sum --check
mv "${partial}" "${target}"
printf 'download_complete %s\n' "${target}"
