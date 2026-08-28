# ICLR / CCF-A claim–evidence gate

## Purpose

This document converts the project story into reviewer-checkable claims. A claim may enter the title, abstract, or introduction only when the listed evidence is complete. Negative results narrow scope; they are not removed from the evidence ledger.

## Proposed paper story

The strongest current story is not “latent q is the best predictor on every scientific dataset.” It is:

> Scientific entity families require a support-conditioned state representation. Explicit latent q is useful when it is trained and calibrated through the same information path, but raw coordinates have a gauge ambiguity that can break downstream symbolic discovery. Decoder-functional gauge fixing makes the representation measurable; cross-batch experiments then test whether the resulting state is predictive, stable, and usable as a compact mechanistic surrogate.

This story has three contributions:

1. a fair support-conditioned latent-variable protocol with exact information accounting;
2. an empirical and dynamical diagnosis of q gauge failure, plus decoder-functional coordinates;
3. a real-data symbolic closed loop, conditional on MATR development and sealed confirmation.

## Claim–evidence map

| Major claim | Required evidence | Current status | Allowed wording now |
|---|---|---|---|
| Support-conditioned latent modeling is materially better than support-blind pooling | Paired synthetic, real, and PDE results against matched no-q MLP; update-count audit | **SUPPORTED** | “Explicit entity adaptation substantially improves over support-blind pooled prediction in the evaluated families.” |
| Prediction-optimal q and representation-optimal q are not identical | Controlled synthetic/Burgers true-factor alignment; PDEBench q-dimension prediction/geometry trade-off | **SUPPORTED** | “Prediction and representation geometry form a reproducible trade-off.” |
| Raw q is unsafe for symbolic discovery because of gauge and calibration-path mismatch | Old Stage C shift/explosion, support-matched diagnostics, Jacobian/manifold analysis, prefix-q experiments | **SUPPORTED, NASA-scoped** | “Arbitrary latent coordinates can be predictive yet fail as transferable symbolic variables.” |
| Decoder-functional coordinates repair raw-q readability | Protocol-matched response projection, physical formula accuracy, entity tails, and cross-seed stability | **SUPPORTED FOR POOLED READABILITY AND STABILITY, NOT FULL TAIL SAFETY**: raw-q ridge is `R²=-1.907461`; scale-aware decoder-functional degree 2 is `0.942488`, minimum decoder fidelity `0.984515`, and functional/raw distance stability `0.835/0.748`, but 9/80 entities still fail the ten-times tail | “Projecting decoder behavior into named equation coefficients repairs coordinate readability; scale-aware training improves entity robustness and cross-seed geometry, while extreme tails remain a limitation.” |
| A stage-wise interpretable real-data expression exists | Entity-OOF/support-query predictions in physical units reach pooled `R² >= 0.85`; the expression uses a stable structure-recalibrated coordinate, has zero query-target leakage, finite predictions, and no catastrophic per-entity failure | **SUPPORTED AND TEMPORALLY CONFIRMED ON STARRY ZT**: development `R²=0.980668`; frozen 30-material, DOI/composition-disjoint confirmation `R²=0.988810`, entity bootstrap `[0.973306,0.994708]`, all six preregistered gates pass | “A frozen three-coordinate response re-parameterization transfers to temporally new, publication- and composition-disjoint ZT materials.” Do not claim uniqueness or predictive superiority over kNN. |
| Functional q improves real scientific prediction over strong support-aware baselines | Fixed methods, exact support, cell-level paired effects on Batch2 and sealed Batch3; kNN/RF/Huber/FPCA included | **NEEDS MATR** | Do not claim yet. |
| q provides additional symbolic value beyond observed support | Symbolic interface beats condition-only and support-anchor baselines with recurring q motif; entity-held-out formula selection | **NOT SUPPORTED on NASA** | “NASA exposes the remaining gauge/selection failure.” |
| A formula-guided decoder yields a closed interpretable loop | First formula → frozen structure → relearn q → second formula; prediction retained and formula/stability improves | **REQUIRED, NOT COMPLETE** | Do not claim completion yet. |
| The method generalizes across real acquisition batches | Batch1→2 development followed by exactly one locked Batch1+2→Batch3 confirmation | **IN PROGRESS** | “We pre-register a cross-batch confirmation protocol.” |

## Reviewer-critical MATR evidence package

The MATR study must answer the five standard top-conference review questions.

### 1. Effectiveness

- Compare no-q MLP, Random Forest, support kNN, robust Huber prefix extrapolation, FPCA/ridge, prefix latent q, and the fixed rank-2 response-prior candidate.
- Use the same cells, protocol features, exact first-100 support, query rows, and reference scale.
- Report per-cell NRMSE, late-life NRMSE, RMSE, MAE, paired ratios, confidence intervals, and failure tails.

### 2. Causality and ablation

- Explicit q versus no q.
- Prefix-q training versus information-mismatched training.
- No functional prior versus fixed rank-2 weight `0.01`.
- Raw q versus decoder-functional `C150` only for representation analysis; raw q must not enter an unsafe symbolic grammar. `C100` is support reconstruction and cannot serve as post-support scientific alignment evidence.

### 3. Strong baselines and fairness

- Neural methods receive 1,000 epochs and matched architecture/update accounting.
- CPU methods are not assigned artificial epochs; their small grids are frozen before Batch2 is summarized.
- Support-aware baselines receive exactly the same first 100 targets.
- Report support kNN even if it wins.

### 4. Harder setting and generalization

- Batch2 is development only.
- Batch3 is a sealed acquisition-batch shift with different rest procedures and cannot tune normalization, method, formula, or exclusions.
- Report protocol subgroups and late-horizon errors; do not hide a bad subgroup by pooling.

### 5. Method soundness

- Machine-audit train/test entity identity and continuation records.
- Verify query-target perturbation leaves q and predictions unchanged until scoring.
- Measure `C150` support-jackknife reliability, cross-seed stability, and post-support empirical alignment before interpreting it.
- Do not start the symbolic/structured stage unless the frozen Batch2 gate authorizes it. Complete all symbolic and structure selection with decoder cross-fitting on Batch1+2, then lock every hash before Batch3 is opened; Batch3 cannot authorize a design choice retroactively.

## Current high-risk rejection points

| Risk | Severity | Required resolution |
|---|---|---|
| Re-q step is simpler than the full neural decoder loop | **High** | The scale-aware decoder-to-coefficient link passes pooled accuracy, fidelity, and distance stability and materially repairs entity medians/tails, but 9/80 still fail the strict ten-times gate. Frame structure re-q as the confirmed endpoint and the neural bridge as supportive with disclosed tails. |
| Strong support-aware methods often beat q on existing real datasets | **Critical** | Obtain a scoped cross-batch win/late-life benefit, or frame q's value around compact transferable state and symbolic use rather than universal accuracy. |
| NASA formula gains are seed- and split-specific | **High** | Do not promote the favorable seeds; use NASA only for failure mechanism unless entity-held-out structure selection changes the frozen conclusion. |
| Functional coordinate has rank stability but insufficient absolute calibration | **High** | Require physical response reliability and cross-seed calibration on MATR before symbolic use. |
| Too many sequential NASA development attempts could look post-hoc | **High** | Freeze NASA; make MATR Batch3 the sole untouched confirmation and expose the full decision ledger. |
| Baseline or compute unfairness | **High** | Preserve exact support, train/update counts, CPU grids, runtime, and all seeds in one table. |

## Acceptance-oriented stopping rule

The mandatory expression endpoint is deliberately narrower than a predictive-superiority claim. It is met by an entity-held-out physical-unit symbolic model with pooled five-seed-median `R² >= 0.85` and the frozen leakage, recurrence, finiteness, and per-battery safety checks. The expression may use a decoder-functional or structure-recalibrated coordinate and need not equal the initial raw `q`, the originally hypothesized physical variable, or a unique ground-truth law. Its scientific value may be stage-wise: a compact recurring relation that suggests a mechanism or the next structural modification.

Beating no-q MLP, kNN, or other support-aware baselines is a separate claim and remains governed by the predictive track. Failure of that track cannot erase a passing interpretable expression, while a passing expression cannot be presented as predictive superiority.

Before describing the work as ICLR-ready, all of the following must be true:

1. MATR eligibility and identity audit passes.
2. A latent candidate passes Batch2 development without changing the frozen gate.
3. A q-derived coordinate passes Batch2 reliability, stability, and scientific-alignment gates.
4. One real symbolic formula meets the mandatory expression endpoint above. This is met on five-fold reviewer-clean Starry ZT development (`R²=0.980668`) and by the frozen, one-shot 30-material temporal confirmation (`R²=0.988810`).
5. Any stronger closed-loop or incremental-value claim separately passes its frozen structure/re-q, comparator, and confirmation gates.
6. All method/formula/structure hashes are frozen, then the same locked package passes the single Batch3 confirmation against no-q and support-aware baselines.
7. An adversarial reviewer pass finds no unsupported abstract/introduction claim and no missing mandatory baseline.

Until then, the honest readiness status is “externally confirmed real expression plus a pooled/stable neural-to-canonical bridge with a disclosed extreme-tail limitation,” not “submission complete.”
