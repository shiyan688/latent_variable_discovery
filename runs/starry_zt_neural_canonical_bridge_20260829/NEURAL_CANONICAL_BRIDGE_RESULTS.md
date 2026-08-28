# Starry ZT neural-to-canonical bridge results

## Question and boundary

This frozen 5-fold x 3-seed development experiment used only the 80 reviewer-clean Starry ZT development entities. It did not read or reuse the 30 temporal-confirmation targets. The scientific question was whether an arbitrary learned neural `q` can be converted through its decoder response into the already frozen, named quadratic coordinates

`ZT(T)=q0+q1*tau+q2*tau^2`.

The expression-existence endpoint is independent and was already passed by support structure re-q. This experiment tests the stronger neural-to-equation method bridge, not whether the expression exists.

## Aggregate result

All 15/15 cells completed, covering 80 held-out entities and 3,879 query rows. The maximum effect of query-target perturbation on model inputs was exactly zero.

| Family | Pooled physical query R² | Physical RMSE | Entity-bootstrap R² 95% interval |
|---|---:|---:|---:|
| raw neural decoder | 0.948354 | 0.065164 | [0.892740, 0.979182] |
| raw-q ridge to quadratic coefficients | -1.907461 | 0.488927 | [-5.238207, 0.109606] |
| decoder-functional degree 1 | 0.942259 | 0.068902 | [0.886622, 0.972955] |
| decoder-functional degree 2 | **0.944683** | **0.067440** | **[0.889918, 0.975222]** |
| decoder-functional degree 3 | 0.948105 | 0.065321 | [0.892612, 0.979030] |
| decoder-functional degree 4 | 0.948464 | 0.065094 | [0.892938, 0.979330] |
| support structure re-q | **0.980668** | **0.039868** | **[0.961729, 0.991593]** |

The primary degree-2 projection exceeds the expression-level `R² >= 0.85` threshold. Its reconstruction of the decoder response is also high in every cell: minimum `R²=0.985033`, median `R²=0.994633`. The failure of raw-q ridge together with the success of decoder-functional projection is direct evidence that raw coordinates are a poor equation vocabulary while the decoder response has a readable quadratic coordinate system.

## Why the full frozen bridge gate does not pass

The aggregate decision remains `neural_to_canonical_bridge_supported=false` because two stronger gates failed.

1. **Entity tail.** The maximum degree-2-functional/structure-re-q entity NRMSE ratio is `744.872`, not at most 10. Nineteen of 80 entities exceed 10; only 40/80 have individual functional degree-2 `R² >= 0.85`, and median entity `R²=0.846748`. This is not merely a small denominator artifact: several near-zero-ZT materials receive neural predictions around `0.01--0.04` when their true responses are about `10^-3`.
2. **Distance stability.** Median raw-q distance-geometry Spearman is `0.794298`, versus `0.738385` for functional coefficients, a change of `-0.055914`. Functional distance geometry is better in only 8/15 seed pairs, so the frozen distance-stability gate fails.

There is a useful but non-gating coordinate result: median unaligned raw-coordinate Spearman is only `0.014148`, while median named functional-coordinate Spearman is `0.733283`; the functional value is higher in all 15/15 seed pairs. Thus canonicalization fixes coordinate naming/readability, but it does not improve every invariant geometry statistic.

## Failure localization

The neural tail is concentrated in low-scale response curves. Across entities, target standard deviation has Spearman `-0.614885` with functional NRMSE and `-0.642429` with the functional/structure error ratio. All five entities with absolute mean ZT below `0.001` fail the ten-times gate; 14 of the 20 entities with absolute mean ZT below `0.02` fail it. These entities are still in-scope ZT curves and are not excluded.

This localizes the next one-factor repair to target-scale weighting: the current label-balanced absolute MSE gives each material similar sampling frequency but does not equalize the scientific cost of errors on low-amplitude curves. A train-fold-only invertible target transform or per-entity scale-normalized loss is the justified next experiment. It must retain all entities and the same folds/support masks.

## Allowed conclusion

The results support the narrower statement that decoder-functional projection converts an unreadable raw-q gauge into a compact named quadratic expression with strong pooled held-out accuracy and near-exact decoder fidelity. They do not support a tail-safe, universally more stable neural-to-canonical bridge. The independently temporally confirmed support structure re-q expression remains valid because it does not depend on this stronger bridge passing.

Authoritative machine-readable outputs are in `analysis/decision.json`, `analysis/family_summary.csv`, `analysis/per_entity_metrics.csv`, and `analysis/cross_seed_stability.csv`.
