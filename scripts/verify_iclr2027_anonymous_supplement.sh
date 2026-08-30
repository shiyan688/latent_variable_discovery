#!/usr/bin/env bash
set -euo pipefail

project_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
archive="$project_root/public/iclr2027_anonymous_supplement_20260830.tar.gz"
clean_root="$project_root/public/qa/iclr2027_supplement_cleanroom_20260830"
receipt="$project_root/public/qa/iclr2027_supplement_verification_20260830.json"
expected_pdf_sha="69b29b6d04375fd683482f7533bf3352ac74b3382551e9f713b5accfdde6ecfc"
expected_archive_sha="484a5bf625e7d20e431b1f4f1a74512d8e63017f870c3cd4588dffdd5bc565ca"

if [[ -e "$clean_root" || -e "$receipt" ]]; then
  echo "clean-room output already exists" >&2
  exit 1
fi
if [[ "$(sha256sum "$archive" | awk '{print $1}')" != "$expected_archive_sha" ]]; then
  echo "archive hash changed" >&2
  exit 1
fi

mkdir -p "$clean_root"
tar -xzf "$archive" --strip-components=1 -C "$clean_root"
(cd "$clean_root" && sha256sum -c SHA256SUMS > prebuild_hash_check.log)
hash_entries="$(wc -l < "$clean_root/SHA256SUMS")"
rm -f "$clean_root/paper/iclr2027_draft.pdf"
(cd "$clean_root" && bash scripts/build_iclr2027_paper.sh > build.stdout.log)

rebuilt_pdf_sha="$(sha256sum "$clean_root/paper/iclr2027_draft.pdf" | awk '{print $1}')"
if [[ "$rebuilt_pdf_sha" != "$expected_pdf_sha" ]]; then
  echo "clean-room PDF is not byte-identical" >&2
  exit 1
fi
if [[ "$(pdfinfo "$clean_root/paper/iclr2027_draft.pdf" | awk '/^Pages:/ {print $2}')" != "13" ]]; then
  echo "unexpected clean-room PDF page count" >&2
  exit 1
fi
pdftotext -f 9 -l 9 "$clean_root/paper/iclr2027_draft.pdf" - | grep -q 'with uncertainty and tails'
pdftotext -f 10 -l 10 "$clean_root/paper/iclr2027_draft.pdf" - | grep -q 'AI USE STATEMENT'

cat > "$receipt" <<EOF
{
  "scope": "isolated anonymous-supplement clean-room verification",
  "archive": "public/iclr2027_anonymous_supplement_20260830.tar.gz",
  "archive_sha256": "$expected_archive_sha",
  "clean_root": "public/qa/iclr2027_supplement_cleanroom_20260830",
  "prebuild_sha256sum_check_exit_code": 0,
  "prebuild_hash_entries": $hash_entries,
  "preexisting_pdf_removed_before_build": true,
  "build_command": "bash scripts/build_iclr2027_paper.sh",
  "build_exit_code": 0,
  "rebuilt_pdf_sha256": "$rebuilt_pdf_sha",
  "expected_source_pdf_sha256": "$expected_pdf_sha",
  "byte_identical": true,
  "pages": 13,
  "page_9_ends_main_text": true,
  "page_10_begins_ai_use": true
}
EOF
echo "verified clean-room receipt: $receipt"
