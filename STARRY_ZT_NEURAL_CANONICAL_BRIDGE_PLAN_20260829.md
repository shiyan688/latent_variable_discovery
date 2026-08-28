# Starry ZT neural-to-canonical q bridge plan

## Question

The frozen expression `ZT(T)=q0+q1*tau+q2*tau^2` is already temporally
confirmed, but its three named coordinates are estimated directly from support.
This development experiment tests the missing method link:

`learned raw q -> decoder response -> canonical equation coefficients -> support re-q`.

The purpose is not to retune the confirmed expression or claim prediction
superiority. It is to determine whether decoder-functional gauge fixing explains
why raw-q symbolic regression failed and supplies a reproducible route from a
black-box latent model to the confirmed equation coordinates.

## Data boundary

Use only the 80 reviewer-clean Starry ZT development entities. Combine the
existing train/test CSVs, sort sample IDs, and assign fold `index mod 5`. The 30
post-snapshot confirmation entities and their targets must not be read by this
experiment. In each outer fold, the other 64 entities are the only training data.
For each held-out entity, sort by temperature; rows `0,4,8,...` are support and
all other rows are query. Query temperatures and composition descriptors are
known covariates; query ZT values are scoring-only.

## Frozen neural protocol

Run seeds `0,1,2` for all five folds. Train a q-dimension-4 Torch MLP with hidden
widths `(256,128)` for 1,000 epochs, batch size 256, Adam learning rate `1e-3`,
label-balanced MSE, HSIC-rich-RFF acquisition orthogonality weight `0.05`, curve
continuity weight `0.05`, and q-L2 weight `0.001`. This matches the strongest old
ZT neural configuration closely enough to diagnose its gauge; it is not selected
against bridge outcomes.

For each held-out entity, freeze the decoder and calibrate raw q from the exact
support only. Use the training-q mean plus three deterministic prior draws as
four starts, 1,200 Adam steps at learning rate `0.05`, q-prior weight `0.01`, and
select the lowest support loss. Query targets never enter calibration or start
selection.

## Fixed bridge analyses

1. **Raw decoder:** evaluate the calibrated raw q with the neural decoder.
2. **Direct raw-q map:** fit a training-entity-only ridge map from raw q to each
   entity's quadratic coefficients. Select alpha by leave-one-entity-out training
   error, then evaluate its held-out formula. This diagnoses whether raw q itself
   has a readable gauge.
3. **Decoder-functional coordinates:** for each entity, probe the frozen decoder
   at 41 temperatures spanning its known covariate interval while holding its
   composition fixed. Project this response onto polynomial degrees 1--4 in the
   fold-standardized temperature `tau`. Evaluate both decoder-response fidelity
   and physical query prediction. Degree 2 is the primary bridge because its
   structure was frozen before the external confirmation; degrees 1,3,4 are
   diagnostics, not alternatives allowed to alter that confirmation.
4. **Structure re-q:** estimate `q0,q1,q2` directly from the same held-out support
   and evaluate the frozen quadratic expression. This is the final gauge-fixed
   coordinate and should reproduce the already-audited development endpoint.

Save a checkpoint, raw q, functional coefficients, per-query predictions,
support masks, training history summary, runtime, and SHA-256 provenance for
every fold-seed cell under `runs/starry_zt_neural_canonical_bridge_20260829/`.
All caches and logs must remain below `runs/`; `/tmp` is forbidden.

## Interpretation gates

The bridge is supported only if all 15 cells finish and query-target input
difference is exactly zero, and the aggregated evidence shows:

- the degree-2 decoder projection has pooled physical query `R^2 >= 0.85`;
- its decoder-response reconstruction `R^2 >= 0.95`;
- structure re-q retains pooled physical query `R^2 >= 0.85`;
- no entity exceeds ten times the structure-re-q NRMSE under the projected
  decoder formula;
- functional coefficient geometry is more cross-seed stable than an unaligned
  raw-coordinate comparison, or the absence of that improvement is reported.

Failure of the direct raw-q map is expected evidence of gauge ambiguity, not a
reason to reject the confirmed expression. Failure of decoder projection would
mean the neural-to-equation bridge is not yet supported; it must not invalidate
the independently confirmed support re-q result. No result here can authorize a
new look at the temporal confirmation targets or MATR Batch3.
