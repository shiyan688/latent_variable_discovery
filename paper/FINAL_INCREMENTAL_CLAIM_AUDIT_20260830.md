# Final incremental claim audit

Date: 2026-08-30

Verdict: **PASS (same-family provisional)**.

This narrow re-review reconsidered only the four process claims left
unsupported by the preceding vapor-selection audit. It did not repeat the
already completed numerical audit.

| Claim | Classification | Evidence |
|---|---|---|
| ZT quadratic endpoint was fixed a priori and not selected against baselines | `exact_match` | The frozen ZT plan declares the quadratic re-q endpoint before evaluation. |
| Supplement was verified from its own root | `exact_match` | The verifier extracts to an isolated persistent root and runs both checksum verification and the build from that root. |
| Every bundled hash passes | `exact_match` | `SHA256SUMS` contains 100 entries and an independent quiet check exits zero; the archive contains those 100 files plus `SHA256SUMS`. |
| The supplement rebuilds a byte-identical manuscript | `exact_match` | The verifier deletes the bundled PDF before compiling; the rebuilt, bundled, and source PDFs all have SHA-256 `69b29b6d04375fd683482f7533bf3352ac74b3382551e9f713b5accfdde6ecfc`. |

Counts: `exact_match=4`, `rounding_ok=0`, `unsupported=0`, `mismatch=0`.
The previous audit's scientific data, configuration, and aggregation findings
remain at zero mismatch; they were not re-audited in this narrow pass.

## Frozen evidence hashes

| Artifact | SHA-256 |
|---|---|
| `STARRY_ZT_INTERPRETABLE_REQ_PLAN_20260829.md` | `f4e4d5b015c46ce398a22b984d4a063609142115fde5de680de978e8d96bcc7f` |
| `scripts/verify_iclr2027_anonymous_supplement.sh` | `33d512b4e1fc1a7c06749094739fb0725ecac489bb42e5dc0f865ad380772d22` |
| `public/qa/iclr2027_supplement_verification_20260830.json` | `73a4a5f6f137cac144bfde103c3306bb70933255696651536805753fce7ca157` |
| `public/iclr2027_anonymous_supplement_20260830/SHA256SUMS` | `656127bb533736ac0208bee14ff13437ed605f6cbbadd5f90163738d62817090` |
| `public/iclr2027_anonymous_supplement_20260830.tar.gz` | `484a5bf625e7d20e431b1f4f1a74512d8e63017f870c3cd4588dffdd5bc565ca` |
| `paper/sections/ai_reproducibility_statements.tex` | `a96ff7dfabc8d4d377dd85b21e1224c6dac6e707994cdcfa71670c67b9f3eeff` |
| `paper/iclr2027_draft.pdf` | `69b29b6d04375fd683482f7533bf3352ac74b3382551e9f713b5accfdde6ecfc` |
