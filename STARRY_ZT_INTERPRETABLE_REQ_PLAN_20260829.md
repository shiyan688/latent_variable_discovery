# Reviewer-clean Starry ZT interpretable re-q plan

## Scope and evidence status

This is a real-data development endpoint on the reviewer-clean StarryData2 ZT
cohort. The prior data audit found zero mixed-property entities among 80 sample
IDs. Earlier exploratory use of this cohort means the result is not an untouched
external confirmation; it is a strict entity-OOF/support-query development test.

## Scientific hypothesis

Within a material, ZT over a measured temperature interval is often described
locally by a low-order response curve. Re-parameterize the entity coordinate as
three physically named coefficients:

`ZT(T)=q0+q1*tau+q2*tau^2`, `tau=(T-mu_train)/sigma_train`.

Here `q0` is the material's reference ZT, `q1` its first-order temperature
sensitivity, and `q2` its curvature. This need not be a universal microscopic
law; it is a stage-wise interpretable response expression.

## Frozen five-fold entity protocol

Assign sorted sample IDs to five folds by index modulo five. In each outer fold,
compute `mu_train` and `sigma_train` from temperatures of the other 64 entities.
For each held-out entity, sort rows by temperature; indices `0,4,8,...` are the
target-free, temperature-stratified support set and all other rows are query.
Fit `q0,q1,q2` by ordinary least squares on support only and score query only.

Report these fixed comparators under the identical query mask:

- linear re-q: `ZT=q0+q1*tau`;
- support kNN: five-neighbor inverse-distance regression in temperature;
- no-q global quadratic fitted from outer-training entities;
- no-q MLP using temperature and composition descriptors, trained only on
  outer-training entities.

The quadratic re-q is the predeclared expression endpoint, not selected against
the baselines. Query-target perturbation must leave every input, q, and prediction
unchanged. Report pooled and per-entity R², physical-unit RMSE, entity bootstrap
confidence interval, q stability over the four possible support offsets, and q
distance versus empirical response-geometry continuity.

## Gate

Development success requires quadratic re-q pooled physical-unit entity-OOF
query R² at least 0.85, finite predictions, exact query-target input invariance,
and no entity above ten times the linear re-q normalized RMSE. Predictive
superiority over support kNN or no-q MLP is explicitly a separate claim.
