#!/usr/bin/env bash
set -euo pipefail

project_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
file_list="$project_root/paper/anonymous_supplement_files.txt"
package_root="$project_root/public/iclr2027_anonymous_supplement_20260830"

if [[ -e "$package_root" ]]; then
  echo "package already exists: $package_root" >&2
  exit 1
fi

mkdir -p "$package_root"
while IFS= read -r relative_path; do
  [[ -z "$relative_path" || "$relative_path" == \#* ]] && continue
  if [[ ! -f "$project_root/$relative_path" ]]; then
    echo "missing allowlisted file: $relative_path" >&2
    exit 1
  fi
  mkdir -p "$package_root/$(dirname -- "$relative_path")"
  cp "$project_root/$relative_path" "$package_root/$relative_path"
done < "$file_list"
cp "$file_list" "$package_root/FILE_INDEX.txt"

if rg -I -n '/public/home|/tmp|wangyg|shiyan688' "$package_root"; then
  echo "machine-local or identifying text found in supplement" >&2
  exit 1
fi

(cd "$package_root" && bash scripts/build_iclr2027_paper.sh >/dev/null)
if [[ "$(pdfinfo "$package_root/paper/iclr2027_draft.pdf" | awk '/^Pages:/ {print $2}')" != "13" ]]; then
  echo "unexpected PDF page count" >&2
  exit 1
fi
pdftotext -f 9 -l 9 "$package_root/paper/iclr2027_draft.pdf" - | grep -q 'with uncertainty and tails'
pdftotext -f 10 -l 10 "$package_root/paper/iclr2027_draft.pdf" - | grep -q 'AI USE STATEMENT'

rm -f \
  "$package_root/paper/iclr2027_draft.aux" \
  "$package_root/paper/iclr2027_draft.bbl" \
  "$package_root/paper/iclr2027_draft.blg" \
  "$package_root/paper/iclr2027_draft.log" \
  "$package_root/paper/iclr2027_draft.out"
rm -rf "$package_root/runs/_runtime_cache"
if rg -I -n '/public/home|/tmp|wangyg|shiyan688' "$package_root"; then
  echo "machine-local or identifying text found after isolated build" >&2
  exit 1
fi

(cd "$package_root" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "verified anonymous supplement: $package_root"
