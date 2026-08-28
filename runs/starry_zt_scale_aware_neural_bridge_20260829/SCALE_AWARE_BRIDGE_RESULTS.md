# Starry ZT scale-aware neural bridge results

## Frozen change

This 15-cell development experiment retained the 80 entities, folds, seeds,
support masks, architecture, q dimension, 1,000 epochs, regularizers, and
calibration budget of the original bridge. Its only scientific change was an
outer-train-only invertible target coordinate:

`z=asinh(y/s_y)`, where `s_y` is the median training-entity population target
standard deviation. Decoder predictions were inverted to physical ZT before
projection and scoring. The temporal-confirmation cohort was not reused.

## Aggregate comparison

| Metric | Original absolute-MSE bridge | Scale-aware bridge |
|---|---:|---:|
| degree-2 functional pooled R² | 0.944683 | 0.942488 |
| degree-2 physical RMSE | 0.067440 | 0.068765 |
| entity-bootstrap R² 95% interval | [0.889918, 0.975222] | [0.884230, 0.974402] |
| minimum decoder-response fidelity R² | 0.985033 | 0.984515 |
| median entity R² | 0.846748 | **0.940261** |
| entities with individual R² >= 0.85 | 40/80 | **52/80** |
| entities above 10x structure-re-q NRMSE | 19/80 | **9/80** |
| maximum functional/structure NRMSE ratio | 744.872 | **139.163** |
| median raw-q distance stability | 0.794298 | 0.748142 |
| median functional distance stability | 0.738385 | **0.835419** |
| median raw unaligned-coordinate stability | 0.014148 | -0.065064 |
| median named functional-coordinate stability | 0.733283 | **0.819800** |

All 15/15 cells completed, all predictions were finite, and the maximum
query-target input difference was exactly zero. Degree-2 pooled physical R²,
decoder fidelity, structure re-q, and functional-over-raw distance stability all
pass. The only failed full-bridge gate is the predeclared maximum ten-times
entity tail, so both `scale_aware_tail_repair_supported` and
`full_neural_to_canonical_bridge_supported` remain false.

## What was repaired

Scale awareness improves functional NRMSE for 59/80 entities. Median functional
NRMSE falls from `0.391448` to `0.244402`, and its median ratio to structure re-q
falls from `3.560720` to `2.358286`. The correlation between target standard
deviation and functional NRMSE weakens from `-0.614885` to `-0.193718`, which
supports the diagnosed scale imbalance rather than an arbitrary lucky gain.

Cross-seed distance geometry also changes from a failed comparison to a passing
one: functional distance Spearman exceeds raw-q distance Spearman in 11/15 pairs
and at the median (`0.835419 > 0.748142`). Named functional coordinates exceed
unaligned raw coordinates in 15/15 pairs.

## Remaining tail

Nine entities still exceed the ten-times ratio. Seven improve relative to the
original bridge, including the four most extreme near-zero curves, but two
higher-scale materials (`6363` and `2100`) worsen. The maximum ratio is still
set by sample `20574`, whose mean/std ZT are about `0.000503/0.000399`; its NRMSE
falls from `36.82` to `6.88`, yet direct support structure re-q is so accurate
that the relative ratio remains `139.16`. No entity is removed or relabeled.

## Interpretation

The one-factor result validates the target-scale diagnosis and makes the neural
bridge substantially more entity-robust and more stable, with only a `0.00220`
pooled R² reduction. It does not meet the deliberately stronger all-entity tail
gate and must not be called a fully supported neural bridge.

Under the separate expression-existence criterion specified by the user—a
compact, scientifically suggestive, strict support-query physical expression
with pooled `R² >= 0.85`, without requiring original-q recovery—the degree-2
functional expression passes comfortably. The independently temporal-confirmed
support structure re-q remains the primary expression evidence.

Authoritative outputs are `analysis/decision.json`, `analysis/family_summary.csv`,
`analysis/per_entity_metrics.csv`, and `analysis/cross_seed_stability.csv`.
