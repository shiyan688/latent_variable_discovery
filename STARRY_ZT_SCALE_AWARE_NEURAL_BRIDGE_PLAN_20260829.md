# Starry ZT scale-aware neural bridge repair plan

## Diagnosis being tested

The frozen neural-to-canonical bridge reached degree-2 functional query
`R²=0.944683` and minimum decoder-response fidelity `R²=0.985033`, but failed
the entity-tail gate. Nineteen of 80 entities exceeded ten times the direct
structure-re-q NRMSE; 14 of those 19 had absolute mean ZT below `0.02`.
Target standard deviation had Spearman `-0.614885` with functional NRMSE.

This experiment tests one diagnosis only: label-balanced sampling did not make
the absolute MSE objective scale-aware, so near-zero ZT response curves were
cheap for the neural decoder to miss. It does not exclude those materials,
change the expression, or seek recovery of the original raw q.

## Frozen information boundary

Use exactly the same 80 reviewer-clean development entities, deterministic five
outer folds, three seeds, temperature-stratified support indices, query masks,
features, architecture, q dimension, 1,000 epochs, regularizers, four calibration
starts, and 1,200 calibration steps as
`STARRY_ZT_NEURAL_CANONICAL_BRIDGE_PLAN_20260829.md`. The 30 temporal-confirmation
entities and targets must not be read. All materials remain in the evaluation.

## Single change

For each outer fold, compute

`s_y = median_l std_population(y_l)`

using outer-training entities only. Train and support-calibrate the neural model
in the invertible target coordinate

`z = asinh(y / s_y)`.

The model's ordinary global target normalization is then applied to `z`, not
physical `y`. Every decoder prediction is returned to physical units by

`y = s_y * sinh(z)`

before response projection or scoring. Direct raw-q-to-coefficient ridge and
support structure re-q remain in physical ZT units. Query targets enter scoring
only. No transform scale or hyperparameter is selected from held-out results.

## Frozen analysis and gates

Run 5 folds x seeds 0--2 and aggregate pointwise median predictions across the
three seeds. Report the same family, per-seed, per-entity, bootstrap, leakage,
decoder-fidelity, and cross-seed stability tables as the original bridge.

The scale-aware repair passes only if:

- all 15 cells succeed and all predictions are finite;
- query-target input difference is exactly zero;
- decoder-functional degree-2 pooled physical query `R² >= 0.85`;
- minimum degree-2 decoder-response reconstruction `R² >= 0.95`;
- support structure re-q physical query `R² >= 0.85`;
- no entity exceeds ten times the structure-re-q NRMSE under degree 2.

The original full bridge additionally required functional distance geometry to
be more cross-seed stable than raw-q distance geometry. That metric is repeated
unchanged and must pass before claiming the full bridge, but it is not silently
redefined as a target-scale repair endpoint. Improvements in named-coordinate
correlation are reported separately.

Passing the pooled expression threshold does not imply prediction superiority,
original-q recovery, a unique physical law, or success on every entity. Failure
does not invalidate the independently temporally confirmed support structure
re-q expression.
