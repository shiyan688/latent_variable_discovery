---
document_type: agent_handoff_memory
project: latent_variable_search
last_updated: 2026-08-30 10:31 CST
current_branch: research/latent-q-stagec-20260826
base_commit: 2b13869
live_status_source: runs/thermoml_crystal_cp_neural_gird_development_20260829/launcher_status.json
---

# Agent memory: latent-variable search

This is the durable handoff record for agents working in this repository. Read it before acting, but do not treat its timestamped live snapshot as current forever. Refresh the referenced JSON and process state first.

## 1. User's research goal and reporting requirements

The project studies support-conditioned latent state discovery for grouped scientific response data. For an unseen entity, a small support set is used to infer a continuous latent vector `q`; a shared decoder then predicts the entity's remaining response curve.

The user expects the work to:

- compare all meaningful method variants, not just one preferred branch/configuration;
- include other methods and a no-latent-variable MLP baseline;
- cover synthetic data, real datasets, and PDEBench rather than extrapolating from synthetic data alone;
- include loss-function ablations, latent-dimension and support-ratio studies;
- quantify q continuity, trustworthiness, neighborhood preservation, distortion, collapse/tear behavior, stability, and physical alignment where ground truth exists;
- report every dataset's performance in tables in the main document;
- explain the task and results so a reader unfamiliar with the work can understand them;
- keep experiments running to completion when authorized, while yielding GPUs that are in use by others.

The frozen 2,910-job extended campaign, full-46 synthetic/recovery work, 230-job 1,000-epoch real/PDE fairness campaign, PDEBench functional baselines, and both q–kNN selector confirmations are complete. Both selectors failed their NASA advancement gate and must not be promoted. The 40-cell support-envelope projected-q confirmation ended infrastructure-affected at 33/40 successes and is not confirmatory. The reviewer-clean NASA 35-cell outer anchor, 30-cell inner q/functional-coordinate audit, and 90-cell inner symbolic Stage C are complete. Stage C passed integrity and broad motif recurrence but failed its downstream-value and readability gates; diagnose and independently freeze an information-matched q interface before any structured-decoder Stage D run.

## 2. Scientific problem in one paragraph

Each entity (for example, a battery, material sample, engine, or PDE parameterization) provides observations `(x, y)`. Training learns a shared predictor `y_hat = f_theta(x, q_entity)` and a low-dimensional entity embedding. At test time, the shared network is frozen and a new entity's `q` is calibrated only from its support observations; query targets remain hidden until scoring. This is few-shot curve completion/system identification, not zero-shot prediction and not a standard probabilistic VAE. A useful q should improve prediction and preserve response geometry, but continuity alone does not prove unique recovery of a physical factor.

## 3. Durable conclusions established before the extended campaign

The authoritative beginner-readable synthesis is [COMPLETE_RESEARCH_REPORT_20260809.md](COMPLETE_RESEARCH_REPORT_20260809.md). Its strongest pre-campaign conclusions are:

- Support/entity information matters: `joint_mse` substantially outperformed the support-blind no-q MLP in the completed synthetic and real paired screens.
- This does not prove an explicit optimized q is necessary. Support kNN was strong on several real datasets, and a learned support-conditioned non-q baseline such as DeepSets/CNP is still an important fairness gap.
- Multi-start, support-internal test-time q calibration was the clearest improvement: the selected q=4 real tasks showed a median NRMSE reduction of 8.29%, with increased calibration cost.
- q=1 was generally too restrictive on the studied real tasks; q=4 improved over q=2 more consistently for alternating optimization, while the joint result was only suggestive.
- Joint and alternating optimization had no universal winner.
- The tested bundled fixed/dynamic regularizers often damaged real-data prediction and local geometry; do not describe those bundles as generally beneficial.
- Prediction-optimal and physically aligned q need not coincide. The controlled Burgers study showed a prediction/identifiability trade-off.
- Continuity must be reported jointly with prediction, cross-seed stability, distortion, collapse/tear rates, and physical alignment where available.

Important chronology: external PDEBench training, its 10-seed sparse-support core, and the later 1,000-epoch matched-update confirmation are complete. Their bounded claims are now incorporated into the main report; raw terminal analysis is under `runs/matched_1000ep_real_pde_20260817/analysis/`.

## 4. Methods and baselines in scope

Latent methods include MSE-only joint and alternating optimization, fixed and dynamically balanced loss bundles, individual loss-component/dose variants, several dependence penalties, q-dimension variants, support-ratio variants, and calibrated multi-start inference.

Required baselines/references include:

- pooled no-q global MLP (support-blind);
- Random Forest (support-blind tabular baseline);
- support-set kNN/local interpolation (support-aware but no learned q);
- true-q-conditioned MLP on synthetic/PDE data as a diagnostic reference, not a deployable method;
- PDE support mean, support kNN, and full-initial-condition PCA-MLP extra-information reference where declared;
- future priority: a learned support-conditioned no-explicit-q baseline such as DeepSets or a Conditional Neural Process;
- completed PDE functional baselines: train-only FPCA with support-only coefficient inference and an information-matched masked DeepONet; FNO is optional only if the paper makes a broad operator-learning claim.

Do not confuse method variants with git branches. As last verified, the only remote branch is `origin/main`. Local `main`, the current `iclr-latent-discovery-pilot` branch, and the remote base all point to commit `2b13869`; the research implementation lives mainly in uncommitted changes.

## 5. Completed extended campaign

### Protocol

- Frozen reservoir: 2,910 deterministic tasks.
- GPUs requested by the campaign: physical devices `2,3,4,5`.
- Current mode: `--run-until-complete`; campaign-level wall-clock deadline removed.
- Per-job safeguard: 90-minute hard timeout remains.
- Availability guard: a new task is dispatched only when observed GPU memory is below 128 MiB.
- Poll interval: 30 seconds.
- Successful task IDs in the append-only ledger are skipped on resume.
- A failure is recorded explicitly. It is not silently retried within the same controller run; because resume skips only successes, a deliberate controller restart would make failed task IDs eligible again.

The protocol and stopping-rule amendment are in [EXTENDED_15H_EXPERIMENT_PLAN_20260809.md](EXTENDED_15H_EXPERIMENT_PLAN_20260809.md).

### Final status

Verified at 2026-08-11 10:17 CST from the terminal campaign state and append-only ledger:

| Field | Value |
|---|---:|
| Planned | 2,910 |
| Completed successfully | 2,910 (100%) |
| Failed | 0 |
| Timed out | 0 |
| Pending | 0 |
| Running | 0 |
| Campaign state | `completed_all` |
| Dispatch deadline | `null` |
| Consolidated result rows | 3,328 |
| Accounted GPU time | 90.40 hours |

The task ledger contains exactly 2,910 rows and 2,910 unique task IDs, all with return code zero. Every reported primary `reference_nrmse` value is finite, and all 3,108 unique referenced `result.json` paths exist. PDE result files can contribute multiple strategy rows, which is why the consolidated table has 3,328 rows rather than one row per dispatched task.

The final task finished at 2026-08-11 03:02:30 CST, the campaign wrote `completed_all` at 03:03:03, and the unlimited takeover completed automatic analysis at 03:03:22. The campaign/takeover tmux session then exited normally. Physical GPUs 2–5 were idle at the 10:17 verification snapshot.

### Full-campaign interpretation

These statements were checked against the consolidated CSVs and their current bounded form is reflected in the main report:

- No single loss variant wins everywhere. The continuity-only preset is the most promising recurring component: against `joint_mse`, its paired median reference-NRMSE delta was -0.0230 over the q-dimension family (65.6% win rate, BH q=0.000461) and -0.0330 over the real support family (56.9% win rate, BH q=0.00461). The explicit dose sweep did not identify a universally significant beneficial weight after correction, so describe this as configuration- and dataset-dependent rather than universal.
- q dimension is dataset-dependent. NASA battery improved strongly as q increased (`joint_mse` median NRMSE 0.766, 0.610, 0.380, 0.231 for q=1,2,4,8). C-MAPSS favored small/moderate q, with `joint_dynamic` best at q=4 (0.00471). Starry Seebeck was dominated by calibration/loss interactions: q=8 continuity reached 0.0188, while many MSE/HSIC variants diverged catastrophically.
- Strong non-latent/support-aware baselines remain essential. On C-MAPSS, Random Forest (about 0.00293) beat the tested latent methods. On Starry Seebeck, support kNN (down to 0.00195) and Random Forest (about 0.0115) beat even the best stable latent configuration; many latent configurations had extreme failure tails. NASA battery was more favorable to the latent continuity model, which beat no-q MLP and was competitive with or better than support kNN at most tested support ratios.
- PDEBench is now genuinely completed. In the 10-seed core, q=16 joint adaptive calibration had median NRMSE 0.2499, beating pooled no-latent MLP (0.9488) and support mean (0.3502) in 10/10 paired seeds. It improved over q=8 adaptive in 9/10 seeds and over q=16 legacy single-start calibration in 10/10 seeds. However, support kNN-4 remained better (0.2113, 10/10 seeds), so the fair claim is that q is useful and competitive, not state-of-the-art among support-aware methods.
- PDE prediction and representation geometry trade off: increasing q from 4 to 16 improved NRMSE but generally reduced continuity AUC while increasing effective rank. This reinforces the earlier conclusion that prediction-optimal and geometry-optimal q need not coincide.

### Runtime sessions and commands

The former campaign tmux session was:

```text
lvs_extended_until_complete_20260810  # completed and no longer present
```

The completed takeover process ran:

```bash
python scripts/run_extended_15h_campaign_20260809.py \
  --gpus 2,3,4,5 \
  --run-until-complete \
  --poll-seconds 30 \
  --single-job-timeout-minutes 90 \
  --output-root runs/extended_15h_campaign_20260809
```

The wrapper [scripts/resume_extended_campaign_until_complete_20260810.py](scripts/resume_extended_campaign_until_complete_20260810.py) waited for the old bounded controller, resumed the same task ledger without a deadline, and successfully launched final analysis after completion.

### Source-of-truth files

| Purpose | Path |
|---|---|
| Live campaign state | `runs/extended_15h_campaign_20260809/campaign_status.json` |
| Frozen manifest | `runs/extended_15h_campaign_20260809/campaign_manifest.json` |
| Frozen task list | `runs/extended_15h_campaign_20260809/planned_tasks.jsonl` |
| Append-only task ledger | `runs/extended_15h_campaign_20260809/task_status.jsonl` |
| Per-task logs | `runs/extended_15h_campaign_20260809/logs/` |
| Takeover state | `runs/extended_15h_campaign_20260809/unlimited_takeover_status.json` |
| Final analysis report | `runs/extended_15h_analysis_20260810/EXTENDED_15H_RESULTS.md` |
| Consolidated result rows | `runs/extended_15h_analysis_20260810/extended_all_result_rows.csv` |
| Group summaries | `runs/extended_15h_analysis_20260810/extended_group_summary.csv` |
| Paired effects | `runs/extended_15h_analysis_20260810/extended_paired_effects.csv` |

The old `analysis_watcher_status.json` may show `analysis_failed`: the bounded watcher raced with the unlimited takeover, saw the newly restarted campaign in `running` state, and exited. This is benign. The unlimited takeover wrapper owns the final analysis and should write a final report after completion.

## 6. How to monitor safely

Read-only checks:

```bash
tmux ls
nvidia-smi
python - <<'PY'
import json
from pathlib import Path
p = Path("runs/extended_15h_campaign_20260809/campaign_status.json")
print(json.dumps(json.loads(p.read_text()), indent=2))
PY
```

Interpretation details:

- Child commands often say `--device cuda:0` because each process receives one physical card through `CUDA_VISIBLE_DEVICES`; this does not mean every job is on physical GPU 0.
- The scheduler currently reaps finished work after its dispatch pass. Short jobs can therefore leave a card idle for up to one 30-second poll interval. This is inefficiency, not necessarily a stall.
- A real stall requires more than a single zero-utilization snapshot: compare successive `updated_at`, `completed`, `pending`, active PIDs, and recent ledger timestamps.
- Do not run a second campaign controller against the same ledger while the takeover session is active.

## 7. Repository state and preservation rules

- Working directory: repository root.
- The campaign originally used a dedicated Python 3.11 GPU environment. Current controllers inherit the interpreter that launches them through `sys.executable`.
- Current branch: `research/latent-q-stagec-20260826`, tracking `origin/research/latent-q-stagec-20260826`.
- Base commit: `2b13869`.
- Remote: `origin`, with only `origin/main` visible at last check.
- The worktree is intentionally dirty with modified/new research code, experiment drivers, analyzers, and tests.

Never run `git reset --hard`, `git checkout -- <path>`, bulk cleaning, or another destructive operation. Existing changes may belong to the user or another agent. Inspect `git status --short` and targeted diffs before editing overlapping files.

Core implementation paths:

- `lvs/core/pipeline.py`: training, calibration, losses, and evaluation;
- `lvs/core/loss_presets.py`: loss configurations;
- `lvs/core/metrics.py`: prediction and latent-geometry metrics;
- `lvs/backends/torch_mlp.py`: default neural backend;
- `scripts/run_iclr_latent_discovery.py`: synthetic discovery jobs;
- `scripts/run_iclr_real_discovery.py`: real-data jobs;
- `scripts/run_pdebench_burgers_latent_study.py`: PDEBench jobs;
- `scripts/run_extended_15h_campaign_20260809.py`: frozen campaign builder/dispatcher;
- `scripts/analyze_extended_15h_campaign.py`: full-campaign consolidation.

## 8. Required completion workflow

Campaign execution and automatic consolidation are complete. Remaining finalization work is:

1. Perform a claim-level audit of every per-dataset and per-expression result, paired effect, uncertainty estimate, continuity/geometry metric, runtime, and calibration failure tail.
2. Diagnose Starry Seebeck's extreme latent failures before treating median-only improvements as robust.
3. Reconcile new results against the pre-campaign conclusions. Do not cherry-pick only favorable loss weights, q dimensions, support ratios, or seeds.
4. Update [COMPLETE_RESEARCH_REPORT_20260809.md](COMPLETE_RESEARCH_REPORT_20260809.md) so every dataset appears in a main-text table and the explanation remains accessible to a new reader.
5. Update this memory's durable-conclusions section after the main-report integration and link any additional audit artifacts.

## 9. Main documents and their roles

| Document | Role |
|---|---|
| `COMPLETE_RESEARCH_REPORT_20260809.md` | Main beginner-readable research report and current claim boundary |
| `EXTENDED_15H_EXPERIMENT_PLAN_20260809.md` | Frozen 2,910-task protocol and no-deadline amendment |
| `NEXT_ROUND_RESEARCH_AND_EXPERIMENT_PLAN_20260809.md` | Proposed optimization and future-domain plan |
| `MATCHED_1000EP_REAL_PDE_PLAN_20260817.md` | Frozen 230-job real/PDE update-matched confirmation protocol |
| `runs/matched_1000ep_real_pde_20260817/analysis/MATCHED_1000EP_REAL_PDE_RESULTS.md` | Terminal audit, per-dataset tables, paired effects, geometry and short-vs-long comparison |
| `PDEBENCH_FUNCTIONAL_BASELINES_PLAN_20260822.md` | Frozen information-matched FPCA/DeepONet protocol |
| `runs/pdebench_functional_baselines_20260822/PDEBENCH_FUNCTIONAL_BASELINE_RESULTS.md` | Terminal audit, method table, paired effects, validation selections, and bounded conclusions |
| `Q_KNN_RELIABILITY_SELECTOR_PLAN_20260822.md` | Frozen support-internal q-versus-kNN selector confirmation now running on new seeds 20--29 |
| `EXPERIMENT_EXECUTION_RECORD_20260808.md` | Earlier experiment provenance and protocol record |
| `LATENT_Q_INTRO_MOTIVATION_THEORY_RELATED_WORK.md` | Theory, motivation, and related-work framing |
| `LATENT_Q_NEURAL_NETWORK_BACKGROUND_AND_THEORY_FOR_STUDENT.md` | Introductory technical explanation |
| `APPLICATION_DATASETS.md` / `REAL_DATASETS2.md` | Dataset provenance and construction |
| `APPLICATION_METHOD_RESULTS_AND_PAPER_ASSESSMENT.md` | Earlier application-focused evidence and leakage audit |

## 10. Open scientific priorities after the frozen campaign

These are proposals, not completed claims:

1. Turn the existing evidence into a frozen paper scope and run a number/claim audit. Preserve the explicit negative results: Starry electrical/thermal neural failures and kNN's PDE advantage.
2. The frozen q–kNN local-global selector confirmation is now active. It must preserve NASA battery accuracy and control all three Starry tails using support-internal validation only; do not tune a selector margin on its formal seed-20--29 cells.
3. FPCA and information-matched DeepONet are complete. Add GP/FNO only if the final paper scope explicitly requires uncertainty or broad operator-learning claims; neither is now the highest-value generic gap.
4. Run at most one frozen confirmatory candidate on new seeds or new in-scope datasets. Report prediction, tail failures, continuity/trustworthiness, distortion, stability, and compute together.
5. Keep scientific curves, inverse problems, parametric PDE/ODE systems, batteries, materials, and digital twins as the primary domain; few-view inverse graphics remains a secondary direction.

## 11. Memory maintenance log

- 2026-08-10 22:50 CST: Created explicit handoff memory and `AGENTS.md` entry point. Recorded the no-deadline takeover, 2,449/2,910 successful tasks, zero failures, shared-GPU contention, current claim boundaries, and required finalization workflow.
- 2026-08-10 22:53 CST: Moved the dirty research worktree from `iclr-latent-discovery-pilot` to the new local branch `research/latent-q-extended-20260810`; no commit or remote push was performed.
- 2026-08-11 10:25 CST: Verified full campaign completion (2,910/2,910, zero failures/timeouts), successful automatic analysis, 3,328 finite result rows, and 90.40 accounted GPU-hours. Recorded preliminary loss, q-dimension, real-baseline, and PDEBench conclusions plus the remaining claim-audit/main-report work.
- 2026-08-11 10:57 CST: Added an ICLR-readiness decision: the experiment volume is sufficient, but the current package is still borderline/weak-reject because the central algorithmic novelty is not yet sharp, learned support-aware baselines are missing, and Starry calibration is unstable. Prioritize one new method plus fair CNP/DeepSets/operator baselines and a frozen confirmatory benchmark rather than more broad loss sweeps.
- 2026-08-11 11:21 CST: Implemented and launched the 54-job exploratory support-encoder pilot on physical GPUs 2–5. The new code adds a fair DeepSets/CNP-style support-conditioned baseline and support-encoder initialization followed by bounded trust-region q refinement. All 3 new unit tests, 35 existing protocol/pipeline tests, and two real-data GPU smoke runs passed. First monitored state: 6/54 success, 0 failures/timeouts, 4 running, 44 pending. No pilot result is yet a durable or confirmatory conclusion.
- 2026-08-11 11:40 CST: Verified terminal support-encoder pilot state: 54/54 success, zero failures/timeouts, 54 unique raw cells, successful automatic analysis, controller exited, GPUs 2–5 released. The encoder-refinement route improved representation continuity but did not produce seed-consistent prediction gains over multistart `joint_continuity`; global-mean DeepSets was too weak. Recorded the next targeted designs (attentive support conditioning and encoder-as-an-extra-candidate) without promoting exploratory results to confirmatory claims.
- 2026-08-11 12:59 CST: Verified terminal targeted follow-up state: 18/18 success with zero failures/timeouts. Encoder-as-an-extra-multistart candidate was often selected but produced no material prediction change; attentive CNP improved over DeepSets only on battery and retained catastrophic Starry tails. Exact capped-data and per-label prediction forensics identified a retained `Bi2Te3=-37.57071` training value versus a capped test range of `[-0.0125835, 0.02059614]`, plus neural extrapolations of order 10^4--10^5 on a few unseen test materials. Frozen the next exploratory hypothesis around support-only target coordinates, robust loss, and bounded standardized residuals rather than another q-initialization sweep.
- 2026-08-11 13:20 CST: Completed the frozen 27-cell support-relative robustness pilot with 27/27 successes, zero failures/timeouts, finite raw predictions, zero bounded-output violations, and successful automatic analysis. The bounded support-relative CNP eliminated all Starry catastrophic tails and reached median reference NRMSE 0.003595, but regressed on battery and C-MAPSS, so the Starry safety gate passed while every general advancement gate failed. Recorded the result as a dataset/regime-specific anti-extrapolation mechanism, not a generally improved method.
- 2026-08-11 13:34 CST: Completed the frozen 9-cell support-internal attentive reliability selector with 9/9 successes, zero failures/timeouts, finite selector scores/predictions, and successful automatic analysis. It improved battery and preserved Starry safety, and the intended routing pattern appeared, but C-MAPSS median NRMSE was 18.4% above the better fixed component, so the preregistered 10% general gate failed. Stopped rather than post-hoc tuning a selector margin on the same development cells.
- 2026-08-11 16:39 CST: After two host-level GPU checks showed physical GPUs 2--5 empty (0, 0, 0, and 4 MiB, with no compute PIDs), launched the 828-job full-library synthetic exploratory matrix in tmux session `lvs_full46_core_20260811`. The matrix covers all 46 supported expressions, six core methods, and exploratory seeds 11--13. Initial health check found 11/828 successful results, zero nonzero return codes, four worker logs, and only project Python processes on the selected cards. This is exploratory breadth evidence, not the frozen confirmatory run.
- 2026-08-11 17:42 CST: Audited cross-method training-budget fairness in the active full-46 matrix. Neural methods share 300 epochs, batch size 256, hidden widths 128/64, and learning rate 1e-3, but the latent joint schedule performs two optimizer/backward updates per batch (4,800 training backward passes) while no-q and true-q MLPs perform one (approximately 2,400 updates). Random Forest uses 200 fixed trees and support kNN has no iterative training. Therefore the matrix is split/information controlled but not strictly update-, compute-, or tuning-budget matched. Preserve it as exploratory breadth evidence and add a frozen matched-update/equal-validation-budget comparison before final confirmatory claims.
- 2026-08-11 17:44 CST: Verified terminal full-46 campaign state: 828/828 atomic results and append-only terminal events, every return code zero, no invalid or missing cell, final `all_runs.csv` and `method_summary.csv` present, and the tmux controller exited normally. Full scientific synthesis remains pending; the completed matrix retains the cross-method compute-budget caveat above.
- 2026-08-11 18:28 CST: Launched a 552-job full-46 neural rerun at 1,000 epochs with optimizer updates matched across neural methods. Added isolated `joint_mse_step1` and `joint_continuity_step1` method names whose latent joint loop performs one rather than two updates per batch; existing methods remain unchanged. A GPU smoke saved a finite success payload with 5/5 expected backward passes, and the first two full cells each recorded 8,000 backward passes. Initial campaign state: 2/552 success, zero nonzero return codes, four project workers on previously empty physical GPUs 2--5. Py-compile and `git diff --check` passed; the GPU environment does not contain pytest, so no new pytest result is claimed for this transition.
- 2026-08-17 10:12 CST: Reconciled the terminal 552/552 matched-update campaign and all Aug-12--14 recovery work. Frozen expression-block paired effects, method/tail/runtime tables, 828/828 recovery-ablation status, 138/138 symbolic-control status, and the strict 28/46 recovery denominator in `runs/matched_update_analysis_20260817/`. The 1,000-epoch no-q MLP did not improve, while both matched-update latent methods beat support-kNN on at least 35/46 expressions.
- 2026-08-17 10:28 CST: Froze and launched the 230-cell 1,000-epoch matched-update real/PDE campaign on newly verified empty physical GPUs 2--5. Added isolated step1 methods to the real/PDE runners and optimizer-counter auditing. GPU smokes passed. Initial live state was 7 successes, zero failures/timeouts, four running, and 219 pending; the availability guard remains active and partial values are not confirmatory.
- 2026-08-17 17:12 CST: Reconciled the terminal matched real/PDE campaign: 230/230 unique jobs succeeded, zero failed or timed out, 200 real and 100 expanded PDE strategy rows have finite primary metrics, both finalizers returned zero, and the controller exited. Froze dataset/seed paired effects, failure tails, geometry, optimizer counters, and short-vs-long comparisons under `runs/matched_1000ep_real_pde_20260817/analysis/`; synchronized the beginner-readable main report, including every in-scope dataset table and the negative findings. The central fairness conclusion is that 1,000 epochs do not remove support-kNN's advantage outside NASA battery; PDE q=16 MSE was worse than its shorter-training anchor in 10/10 seeds. Final host snapshot found no tmux sessions; GPUs 0--5 were occupied by other workloads and GPUs 6--7 were idle, with no completed campaign process to resume.
- 2026-08-22 00:22 CST: Froze and launched the 20-cell PDEBench functional-baseline campaign in tmux session `lvs_pde_functional_20260822` after three host checks found physical GPUs 4--5 empty. The matrix adds train-only FPCA with support-only ridge coefficient inference and a masked sparse-support DeepONet with a 128,000-update cap matched to the existing q=16 neural backward count. CPU smoke, GPU smokes, exact 11,488-row query alignment, query-target perturbation, py-compile, and `git diff --check` passed. Initial live state: 10/10 FPCA cells successful, zero failures/timeouts, DeepONet seeds 0--1 running on GPUs 4--5, and eight DeepONet cells pending. Partial values are not confirmatory.
- 2026-08-22 00:50 CST: Reconciled the terminal PDEBench functional-baseline campaign: 20/20 unique jobs succeeded, zero failed or timed out, every result has 11,488 exact matched query rows, query-target perturbation changes predictions by 0, automatic analysis returned zero, and the controller exited. FPCA median NRMSE is 0.2753, masked DeepONet 0.3136, q=16 continuity 0.2573, and support-kNN 0.2113. q beats FPCA and DeepONet in 10/10 paired seeds but loses to kNN in 10/10. GPUs 4--5 were empty after completion; GPUs 0--3/6--7 belonged to unrelated workloads.
- 2026-08-22 01:03 CST: Froze and launched the 40-cell q–kNN support-internal reliability-selector confirmation on newly rechecked empty GPUs 4--5. It covers NASA battery and the three in-scope Starry properties on new seeds 20--29. A 5-epoch NASA GPU smoke passed exact query/selector/q-row checks, finite-schema checks, and a zero query-target leakage probe. Initial formal state: 0 complete, 0 failed, 38 pending, NASA seed 20 on GPU 4 and Starry Seebeck seed 20 on GPU 5; both cards contain only the new project workers.
- 2026-08-25 16:37 CST: Reconciled the reviewer-clean NASA inner matrix at 30/30 successful q cells and 30/30 completed frozen-decoder functional analyses. Continuity was worse than MSE on prediction in 11/15 paired cells but reproduced near-perfect cross-seed q-distance geometry in all three inner splits. Frozen cycle-1 capacity and early fade as the primary functional vocabulary for the pending same-budget symbolic comparison; synchronized the run report, main report, milestone, and memory without launching symbolic regression.

## 12. ICLR readiness gate

This is a project-level quick assessment, not a full review of a finished ICLR manuscript. Current verdict: **experimentally substantial but not yet submission-robust; likely Weak Reject / Borderline at ICLR without further work**.

Strengths already at ICLR scale:

- 2,910 successful jobs, multiple seeds, synthetic/real/PDE domains, paired effects, BH correction, runtime accounting, and representation-geometry metrics;
- an interesting empirical message that support information matters while explicit q is not universally superior;
- a reproducible prediction/continuity/identifiability trade-off on controlled Burgers and PDEBench tasks.

Hard blockers:

1. The central algorithmic contribution is not yet sufficiently distinct from auto-decoder/test-time latent optimization, meta-learning, neural processes, and system identification. Loss recipes and broader evaluation alone may look incremental.
2. DeepSets, attentive CNP, support-relative bounded CNP, a reliability selector, FPCA, and information-matched masked DeepONet have now been run. They substantially close the missing-baseline gap, but none establishes a generally superior support-to-latent method; the remaining scientific blocker is a robust candidate that preserves battery accuracy while controlling Starry tails.
3. Starry Seebeck has catastrophic calibration tails across many latent configurations. Best-cell medians are not submission-safe until failure rate, scaling, and robust calibration are diagnosed.
4. Broad sweeps create model-selection risk. Any promoted method must be selected on a development subset and confirmed once on frozen held-out datasets/seeds without post-hoc tuning.
5. The dirty worktree and local-only results need an immutable code commit, environment lock, deterministic manifests, and a clean reproduction command before submission.

Minimum go/no-go experiments before an ICLR submission:

1. The first permutation-invariant encoder-plus-refinement pilot did not pass the go/no-go gate. The next single crisp method should preserve the existing multistart fallback and add the encoder output as an extra support-internally selected candidate; do not launch an unconstrained architecture sweep.
2. Compare it fairly against the completed optimized-q, DeepSets/CNP, support kNN, Random Forest, FPCA, and masked DeepONet anchors, with matched information and declared tuning budgets. Add GP/FNO only if required by the final claim scope.
3. Freeze the selected method and run a confirmatory matrix across all declared synthetic and real datasets plus PDEBench, reporting prediction, tails/failures, support-size generalization, geometry/stability, and compute.

The strategic paper story should be either a genuinely improved support-to-latent inference method, or a rigorous benchmark/diagnostic paper answering when an explicit latent bottleneck helps. Do not sell the current package as universally state of the art.

## 13. Completed support-encoder pilot (exploratory)

Launched at 2026-08-11 11:20 CST in tmux session `lvs_support_encoder_pilot_20260811` after verifying physical GPUs 2–5 were empty. This campaign directly addresses ICLR readiness blockers 1–2 but is explicitly a development pilot, not a frozen confirmatory run.

Terminal state verified at 2026-08-11 11:40 CST:

| Field | Value |
|---|---:|
| Planned / successful | 54 / 54 |
| Failed / timed out | 0 / 0 |
| Unique reconciled raw cells | 54 |
| Analysis return code | 0 |
| Controller | exited normally |
| GPUs 2–5 after completion | 0, 0, 0, 4 MiB |

### Frozen matrix and implementation

- Datasets: NASA battery capacity (q=8), NASA C-MAPSS FD001 sensor response (q=4), Starry Seebeck (q=8).
- Methods: `deepsets_direct`, `encoder_q_refine`, `joint_continuity`, `no_q_mlp`, `random_forest`, `support_knn`.
- Seeds: 0, 1, 2; support ratio 0.3; 256-row per-label train/test caps; 200 training epochs.
- `encoder_q_refine`: permutation-invariant support encoder, episodic query reconstruction plus weak train-q alignment, then 50 support-only refinement steps with trust-region weight 0.01 and train-q coordinate clipping at 3 standard deviations.
- Scheduler: four physical GPUs, 30-second polling, <128 MiB dispatch guard, 90-minute per-job timeout, no campaign deadline, no automatic retry after a terminal failure/timeout.

Code and protocol:

- `lvs/backends/support_conditioned.py`
- `scripts/run_support_conditioned_real_study.py`
- `scripts/run_iclr_support_encoder_campaign_20260811.py`
- `scripts/analyze_support_encoder_pilot_20260811.py`
- `tests/test_support_conditioned.py`
- `ICLR_SUPPORT_ENCODER_PILOT_PLAN_20260811.md`

Source-of-truth runtime files:

| Purpose | Path |
|---|---|
| Live state | `runs/iclr_support_encoder_pilot_20260811/campaign_status.json` |
| Frozen manifest | `runs/iclr_support_encoder_pilot_20260811/campaign_manifest.json` |
| Planned jobs | `runs/iclr_support_encoder_pilot_20260811/planned_tasks.jsonl` |
| Append-only events | `runs/iclr_support_encoder_pilot_20260811/task_events.jsonl` |
| Per-job logs | `runs/iclr_support_encoder_pilot_20260811/job_logs/` |
| Raw new-method results | `runs/iclr_support_encoder_pilot_20260811/new_methods/` |
| Raw anchor results | `runs/iclr_support_encoder_pilot_20260811/anchors/` |
| Terminal analysis (when complete) | `runs/iclr_support_encoder_pilot_20260811/SUPPORT_ENCODER_PILOT_RESULTS.md` |

### Exploratory outcome

Use reference-scaled NRMSE to read the following medians; Starry macro NRMSE is numerically dominated by near-constant per-label scales. The full report retains both metrics and every seed range.

| Dataset | encoder q + refine | DeepSets direct | joint continuity | Best tested method |
|---|---:|---:|---:|---|
| NASA battery | 0.487 | 0.609 | 0.218 | joint continuity (0.218) |
| C-MAPSS | 0.007 | 0.078 | 0.007 | Random Forest (0.003) |
| Starry Seebeck | 0.160, range 0.058–28976.707 | 27584.126 | 0.030, range 0.005–28976.564 | support kNN (0.002) |

The learned q route did not pass the prediction go/no-go gate: versus `joint_continuity`, paired reference-NRMSE wins were 0/3 on battery, 1/3 on C-MAPSS, and 0/3 on Starry. It did consistently improve the separate representation-continuity comparison (3/3, 3/3, and 2/3 seed wins), with lower collapse on battery and Starry. This is a prediction/geometry trade-off, not a universal gain.

Refinement improved the encoder's own initialization in 3/3 battery seeds, 1/3 C-MAPSS seeds, and nominally 3/3 Starry seeds. However, Starry seed 2 remained catastrophic for both `joint_continuity` (reference NRMSE 28976.564) and encoder refinement (28976.707), even after support loss fell sharply. That localizes the main failure to decoder/data-tail stability and support-to-query mismatch rather than q initialization alone.

Global-mean DeepSets was worse than `joint_continuity` in all paired prediction cells and collapsed on Starry. The next justified learned support baseline is query-to-support attention/local conditional aggregation. The next justified q method is encoder-as-an-extra-candidate inside the existing support-internal multistart selector. Neither should be called confirmatory until selected on development data and rerun with a separately frozen seed/dataset matrix.

## 14. Targeted attentive/multistart follow-up and Starry forensics (exploratory)

Terminal state verified at 2026-08-11 12:59 CST. The tmux controller exited normally, all 18 raw jobs succeeded, automatic analysis returned zero, and physical GPUs 2--5 were free at the verification snapshot.

### Frozen matrix and outcome

- New methods: `attentive_cnp` and `encoder_q_multistart`.
- Datasets/seeds: the same three real datasets and seeds 0--2 as section 13; the earlier anchors were reused without rerunning them.
- `encoder_q_multistart` computed the encoder candidate only from the calibration-internal fit rows, then compared it with the four existing prior-random candidates on the existing internal selection subset before final refinement. Query targets were not exposed.
- `attentive_cnp` used query-to-support cross-attention and a permutation-invariant global support representation.

| Dataset | encoder q multistart | joint continuity anchor | attentive CNP | DeepSets anchor | support kNN anchor |
|---|---:|---:|---:|---:|---:|
| NASA battery | 0.232 | 0.218 | 0.462 | 0.609 | 0.304 |
| C-MAPSS | 0.007 | 0.007 | 0.196 | 0.078 | 1.136 |
| Starry Seebeck | 0.030 (0.005--28976.563) | 0.030 (0.005--28976.564) | 4946.967 (0.173--78674.200) | 27584.126 | 0.002 |

Values are median reference-scaled NRMSE over three seeds; parentheses show the Starry seed range where tails are essential. Both preregistered development gates failed:

- Encoder multistart beat the anchor in 1/3 battery, 2/3 C-MAPSS, and 1/3 Starry seeds, but deltas were numerically negligible and the shared Starry seed-2 decoder failure remained. The encoder candidate selection fractions were nonzero (median approximately 0.444, 0.120, and 0.150 by dataset), so the null result is not explained by the candidate being ignored.
- Attentive CNP beat DeepSets in 3/3 battery, 0/3 C-MAPSS, and 1/3 Starry seeds. It therefore did not establish a generally stronger learned support baseline. Starry representations remained close to rank one and locally collapsed.

Source-of-truth files:

| Purpose | Path |
|---|---|
| Frozen protocol | `ICLR_SUPPORT_FOLLOWUP_PLAN_20260811.md` |
| Campaign status | `runs/iclr_support_followup_20260811/campaign_status.json` |
| Raw new results | `runs/iclr_support_followup_20260811/new_methods/` |
| Combined seed-level rows | `runs/iclr_support_followup_20260811/combined_results.csv` |
| Paired effects | `runs/iclr_support_followup_20260811/paired_effects.csv` |
| Human-readable report | `runs/iclr_support_followup_20260811/SUPPORT_FOLLOWUP_RESULTS.md` |

### Starry failure localization

The production runner caps each label at 256 rows with fixed seed 20260808. Reproducing that exact cap gives 13,812 training rows and 4,745 test rows. The capped training target has mean -0.001757, standard deviation 0.319898, minimum -37.57071, median 7.55e-06, and maximum 0.257662. The capped test target is far narrower: mean -0.000154, standard deviation 0.001347, minimum -0.0125835, and maximum 0.0205961. A single retained `Bi2Te3` row dominates the global target scale.

The failures are not diffuse metric artifacts. On catastrophic seeds, neural models extrapolate nearly constant predictions of order 10^4--10^5 for a few unseen test materials (especially `SiC` and `Sm1.85Ce0.15CuO4`) even though their true responses are near zero. Support kNN stays within the observed support scale and is stable across all three seeds. This evidence supports a targeted next ablation:

1. express each episode's targets in support-only robust coordinates (support median and MAD with a training-only floor);
2. compare MSE with SmoothL1/Huber in those coordinates;
3. test a bounded standardized residual head as an explicit anti-extrapolation mechanism;
4. evaluate in untouched physical target units and retain all tail/failure reporting.

This is an exploratory robustness experiment, not a confirmatory claim. Statistics for a test material must use only its support rows; query targets may be used only for final metrics.

## 15. Completed support-relative robustness pilot (exploratory)

Terminal state verified at 2026-08-11 13:20 CST:

| Field | Value |
|---|---:|
| Planned / successful | 27 / 27 |
| Failed / timed out | 0 / 0 |
| Unique raw cells | 27 |
| Finite raw query predictions | 27 / 27 |
| Bounded-output violations | 0 |
| Automatic analysis return code | 0 |
| Controller | exited normally |

The frozen protocol compared support-relative robust target coordinates with MSE, with SmoothL1, and with SmoothL1 plus an 8-standardized-unit tanh bound. All variants used only the current entity's support targets for its median/MAD coordinates and evaluated untouched query targets in physical units.

| Dataset | support-norm MSE | support-norm Huber | support-norm Huber + bound | original attentive CNP | support kNN |
|---|---:|---:|---:|---:|---:|
| NASA battery | 1.118 (0.960--4.449) | 0.637 (0.554--2.249) | 0.902 (0.251--0.936) | 0.462 | 0.304 |
| C-MAPSS | 0.387 (0.384--0.450) | 0.860 (0.842--0.877) | 0.506 (0.489--0.535) | 0.196 | 1.136 |
| Starry Seebeck | 1125 (132--3943) | 21.83 (18.04--63.60) | 0.003595 (0.003510--0.003989) | 4947 (0.173--78674) | 0.002178 |

Values are median reference-scaled NRMSE with the three-seed range in parentheses. The evidence separates the mechanisms:

- Support-relative normalization alone did not prevent extrapolation. On Starry its standardized predictions still reached approximately 1.90e7.
- SmoothL1 reduced but did not eliminate the tail; all three Starry seeds remained catastrophic and standardized outputs reached approximately 6.82e5.
- The explicit bound eliminated the Starry tail in 3/3 seeds, constrained physical predictions to roughly the observed support scale (maximum absolute predictions 0.00546--0.00688), and beat Random Forest and the median latent anchor, although support kNN remained better.
- The same bound discarded useful amplitude on battery and C-MAPSS. It lost to the original attentive CNP in 2/3 battery and 3/3 C-MAPSS seeds. Consequently the preregistered Starry safety gate passed, but all general advancement gates failed.

The durable interpretation is that bounded support-relative output is an effective **failure-control mechanism for extreme scale shift**, not a universal replacement for global target coordinates. A justified next method is a support-internal reliability selector or mixture that chooses between the expressive global attentive head and the safe bounded support-relative head without using query targets. This proposal requires a new frozen protocol.

Source-of-truth files:

| Purpose | Path |
|---|---|
| Frozen protocol | `ICLR_SUPPORT_RELATIVE_ROBUSTNESS_PLAN_20260811.md` |
| Campaign state | `runs/iclr_support_robustness_20260811/campaign_status.json` |
| Raw results | `runs/iclr_support_robustness_20260811/new_methods/` |
| Combined rows | `runs/iclr_support_robustness_20260811/combined_results.csv` |
| Paired effects | `runs/iclr_support_robustness_20260811/paired_effects.csv` |
| Frozen gate decisions | `runs/iclr_support_robustness_20260811/gate_decisions.json` |
| Human-readable report | `runs/iclr_support_robustness_20260811/SUPPORT_ROBUSTNESS_RESULTS.md` |

At the terminal GPU snapshot, physical GPUs 3--5 had returned to 4 MiB. GPU 2 had subsequently been occupied by an unrelated process (about 37 GiB); do not use it without a fresh availability check.

## 16. Completed attentive reliability-selector pilot (exploratory)

Terminal state verified at 2026-08-11 13:34 CST: 9/9 jobs succeeded, zero failed or timed out, all raw predictions and both per-entity selector scores were finite, automatic analysis returned zero, and the tmux controller exited normally. Physical GPUs 3--5 returned to 4 MiB; GPU 2 remained occupied by another user.

The selector trains the frozen global attentive CNP and bounded support-relative attentive CNP. For each test entity, it partitions only the external support rows into selector-fit and selector-validation, chooses the lower physical-unit MAE component, then reuses all external support rows to predict untouched query rows. Query-target perturbation tests leave choices and predictions unchanged.

| Dataset | selector reference NRMSE | global component | bounded component | bounded selected |
|---|---:|---:|---:|---:|
| NASA battery | 0.3627 (0.2525--0.4421) | 0.4619 | 0.9015 | 0.7778 |
| C-MAPSS | 0.2318 (0.1928--0.2688) | 0.1958 | 0.5063 | 0.1200 |
| Starry Seebeck | 0.003595 (0.003508--0.003634) | 4947 | 0.003595 | 0.9500 |

The validity and Starry safety gates passed. The routing mechanism was qualitatively supported: bounded selection was highest on Starry and low on C-MAPSS. The selector beat global attentive CNP in 3/3 battery and 3/3 Starry seeds, but in 0/3 C-MAPSS seeds. Its C-MAPSS median was 18.4% above the better component, exceeding the frozen 10% tolerance; therefore the general advancement gate failed.

Do not tune a selector margin post hoc on these same nine cells. A next selector should estimate a conservative uncertainty/margin rule using training entities or a separately declared development split, then be assessed on held-out seeds/datasets. The current selector is promising exploratory evidence, not a promoted or confirmatory method.

Source-of-truth files:

| Purpose | Path |
|---|---|
| Frozen protocol | `ICLR_ATTENTIVE_RELIABILITY_SELECTOR_PLAN_20260811.md` |
| Campaign state | `runs/iclr_attentive_selector_20260811/campaign_status.json` |
| Raw results | `runs/iclr_attentive_selector_20260811/new_method/` |
| Combined rows | `runs/iclr_attentive_selector_20260811/combined_results.csv` |
| Paired effects | `runs/iclr_attentive_selector_20260811/paired_effects.csv` |
| Gate decisions | `runs/iclr_attentive_selector_20260811/gate_decisions.json` |
| Human-readable report | `runs/iclr_attentive_selector_20260811/ATTENTIVE_SELECTOR_RESULTS.md` |

## 17. Active full-46 synthetic exploratory campaign

Launched at 2026-08-11 16:37 CST after a corrected deterministic generation audit found all 46 current library records syntactically supported and able to produce finite, nonzero-variance disjoint-label samples. Their ground-truth latent dimensions are distributed as 23 one-dimensional, 16 two-dimensional, and 7 three-dimensional tasks. Several tasks have extreme target dynamic range (especially expression 41), so final analysis must retain per-expression tails and failure counts rather than relying on a pooled median.

Runtime contract:

| Field | Value |
|---|---|
| tmux session | `lvs_full46_core_20260811` |
| output root | `runs/full46_synthetic_core_exploratory_20260811/` |
| expressions | all 46 supported IDs in `data/latent_variable_expressions.csv` |
| methods | `joint_mse`, `joint_continuity`, `no_q_mlp`, `random_forest`, `support_knn`, `oracle_q_mlp` |
| seeds | 11, 12, 13; fixed data seed 20260811 |
| planned jobs | 46 x 6 x 3 = 828 |
| physical GPUs | 2, 3, 4, 5 |
| protocol | disjoint 32 train / 16 validation / 32 test labels; 60 samples per label; support ratio 0.3; 300 epochs and 300 calibration steps |
| campaign deadline | none; launcher runs until the queue drains |

The existing launcher resumes exact successful cells and writes an append-only `launcher_status.jsonl`, but it has no live GPU-availability guard and no per-job hard timeout. Do not launch a second controller against the same output root. Monitor the tmux session, result count, status return codes, log growth, and GPU process paths. A GPU may be temporarily idle when its assigned job is a CPU-only baseline such as Random Forest or support kNN.

Source-of-truth files:

| Purpose | Path |
|---|---|
| Frozen runtime manifest | `runs/full46_synthetic_core_exploratory_20260811/experiment_manifest.json` |
| Append-only terminal events | `runs/full46_synthetic_core_exploratory_20260811/launcher_status.jsonl` |
| Per-job logs | `runs/full46_synthetic_core_exploratory_20260811/logs/` |
| Atomic results | `runs/full46_synthetic_core_exploratory_20260811/expr*/**/result.json` |
| Terminal aggregate (after completion) | `runs/full46_synthetic_core_exploratory_20260811/all_runs.csv` and `method_summary.csv` |

Initial verified observation at 2026-08-11 16:38 CST: 11/828 result files, 11 zero-return-code terminal events, zero recorded failures, and the controller alive. Terminal state verified at 2026-08-11 17:44 CST: 828/828 atomic results, 828/828 zero-return-code events, zero failures, both aggregate CSVs present, and the controller exited normally. Scientific interpretation has not yet been frozen.

Cross-method fairness caveat verified during the run: all neural rows record 300 epochs, but `joint_steps_per_cycle=2` gives the two latent methods 4,800 training backward passes versus approximately 2,400 optimizer steps for the no-q and true-q MLPs. Random Forest instead fits 200 trees (`min_samples_leaf=2`, all features), and support kNN fits distance-weighted `k<=5` neighbors separately within each held-out entity. Thus equal epoch labels do not imply equal compute. The final confirmatory protocol must (1) match neural optimizer-update counts in a dedicated ablation, (2) give every tunable method the same train-entity validation trial budget, and (3) report wall time and performance-versus-compute rather than pretending that epochs apply to RF or kNN.

## 18. Completed 1,000-epoch matched-neural-update rerun

The user requested an epoch-1000 rerun after the compute audit showed that the exploratory 300-epoch latent methods used twice as many per-batch updates as the no-q and true-q MLPs. Random Forest and support kNN have no epoch loop, so their completed same-split rows from section 17 are reused as fixed anchors rather than rerun cosmetically.

Implementation and runtime contract:

| Field | Value |
|---|---|
| tmux session | `lvs_full46_1000ep_matched_20260811` |
| output root | `runs/full46_synthetic_neural_1000ep_matchedupdates_20260811/` |
| expressions | all 46 supported expressions |
| methods | `joint_mse_step1`, `joint_continuity_step1`, `no_q_mlp`, `oracle_q_mlp` |
| seeds / data seed | 11, 12, 13 / fixed 20260811 |
| planned jobs | 46 x 4 x 3 = 552 |
| epochs | 1,000 for every neural method |
| neural update budget | approximately 8,000 optimizer/backward steps per job for every method |
| physical GPUs | 2, 3, 4, 5; confirmed empty immediately before launch |
| other protocol fields | identical to section 17: 32/16/32 disjoint labels, 60 samples per label, support ratio 0.3, batch size 256, 300 test-time calibration steps for latent methods |

The new method names change only `joint_steps_per_cycle` from 2 to 1 and preserve all existing method definitions. The runtime manifest now serializes complete method configs, including this field. A 5-epoch GPU smoke for `joint_mse_step1` returned success, finite NRMSE, and exactly 5 backward/theta/q steps. Initial full-run verification at 2026-08-11 18:27 CST found 2/552 successes, zero failures, and exactly 8,000 backward passes for both completed latent cells.

This experiment tests whether longer, update-matched neural optimization closes the gap to fixed RF/kNN anchors. It still does not equalize hyperparameter-search budgets across algorithm families. Final interpretation must compare (a) 300-epoch original neural rows, (b) 1,000-epoch matched-update rows, and (c) the unchanged CPU anchors on identical expression/seed/data blocks, while reporting runtime and tails.

Terminal observation (2026-08-11 20:25 CST, reconciled 2026-08-17 10:10 CST): all 552/552 jobs finished successfully, all 552 launcher return codes are zero, all `reference_nrmse` values are finite, and every latent job records exactly 8,000 backward/theta/q steps. The controller exited normally. The terminal aggregate files are `all_runs.csv` and `method_summary.csv` under the output root above.

Durable paired interpretation, frozen on 2026-08-17 in `runs/matched_update_analysis_20260817/`:

- The 1,000-epoch no-q MLP did not improve over its 300-epoch counterpart: median NRMSE 0.7045 versus 0.6991, with expression-level wins on only 12/46 tasks. More epochs alone do not explain the no-q deficit.
- `joint_mse_step1` reached median NRMSE 0.01450 and beat the 300-epoch/two-step `joint_mse` on 33/46 expressions. `joint_continuity_step1` reached 0.02204 and also won on 33/46 expressions. Because total updates changed from about 4,800 to 8,000, these are final long-training comparisons, not pure one-factor epoch ablations.
- Against the fixed same-split support-kNN anchor (median 0.1431), `joint_mse_step1` won on 36/46 expressions and `joint_continuity_step1` on 35/46. The median expression-level NRMSE deltas were -0.0982 and -0.0784, respectively. The no-q MLP won on only 6/46 expressions.
- Random Forest and kNN remain fixed-complexity anchors with no epoch count. Do not describe their comparison as epoch-matched; only the neural optimizer-update comparison is matched.

Source-of-truth analysis artifacts:

| Purpose | Path |
|---|---|
| Beginner-readable synthesis | `runs/matched_update_analysis_20260817/MATCHED_UPDATE_AND_RECOVERY_RESULTS.md` |
| Method/tail/runtime table | `runs/matched_update_analysis_20260817/method_summary.csv` |
| Expression-block paired effects | `runs/matched_update_analysis_20260817/paired_effects.csv` |
| Per-expression medians | `runs/matched_update_analysis_20260817/expression_method_medians.csv` |

## 19. Completed all-46 latent-recovery and symbolic-control audit

Timestamped terminal observations, reconciled 2026-08-17 10:12 CST:

- `runs/q_recovery_ablation_20260812/` contains 828/828 atomic results (46 expressions x 6 methods x 3 seeds), with 828 zero launcher return codes. The variants were original joint continuity, fixed-norm, affine-quotient, alternating q-LR x1/x10, and a gradient-logging duplicate/control.
- `runs/symbolic_all46_20260814/` contains 138/138 successful symbolic jobs: 46 expressions x learned-q/no-q/entity-one-hot, seed 13, disjoint fit/held-out labels.
- Final recovery accounting is frozen in `runs/recovery_final_20260814/recovery_final.csv` and `recovery_final_summary.json`. The denominator is all 46 expressions, with no sensitivity exclusion: 28 recovered (60.9%), 12 not recovered, 3 optimization-diverged, and 3 weak-control-margin.
- Learned-q symbolic held-out R2 has median 0.9829, versus 0.0564 for no-q and 0.0473 for entity one-hot. Three learned-q symbolic fits nevertheless diverged catastrophically, so pooled means are not a defensible summary.

Durable conclusions:

- The all-46 controls support learned q as a useful interface for downstream symbolic discovery, but the strict 28/46 rate is not universal recovery.
- None of the recovery optimization variants is a general replacement for the original joint-continuity method. The original method has the lowest global prediction-NRMSE median (0.01968); alternating q-LR x1/x10, fixed-norm, and affine quotient do not win consistently. Fixed-norm and affine quotient substantially worsen prediction and representation metrics.
- Preserve the fixed denominator and failure taxonomy in the paper. Denominator structures, multi-q interactions, numerical overflow, and weak control margin are material failure modes, not rows to silently exclude.

## 20. Completed 1,000-epoch matched-update real/PDE campaign

This campaign was frozen and launched on 2026-08-17 to answer the remaining fairness question on the domains where support-kNN was strongest. It excludes the dataset the user removed from manuscript scope.

| Field | Value |
|---|---|
| Frozen plan | `MATCHED_1000EP_REAL_PDE_PLAN_20260817.md` |
| tmux session | `lvs_matched1000_real_pde_20260817` |
| campaign root | `runs/matched_1000ep_real_pde_20260817/` |
| GPUs | physical 2, 3, 4, 5 only; each verified at 0 MiB immediately before launch |
| real cells | 200: NASA battery plus three separate Starry properties x 10 seeds x 5 methods |
| real methods | `joint_mse_step1`, `joint_continuity_step1`, `no_q_mlp`, fixed support-kNN, fixed Random Forest |
| real protocol | q=8, 1,000 epochs, one update per batch for neural methods, adaptive four-start calibration, support ratio 0.3, capped 256 rows/label |
| PDE cells | 30: q=16 step1 MSE/continuity x 10 seeds, plus q=8 step1 MSE baseline block x 10 seeds |
| campaign total | 230 atomic jobs, no wall-clock deadline, 240-minute per-job timeout |
| dispatch guard | dispatch only below 128 MiB observed memory; wait if another user occupies a selected card |

Implementation changes add isolated `joint_mse_step1` and `joint_continuity_step1` method configurations to the real and PDE runners without changing existing method definitions. The PDE payload now records latent and baseline optimizer counters; the real no-q payload records its inferred exact optimizer/backward counts. Py-compile, `git diff --check`, configuration-level assertions, and GPU smokes passed. The project GPU environment still lacks pytest; the system pytest environment lacks torch, so no pytest pass is claimed.

Terminal observation: `campaign_status.json` reached `completed_all` at 2026-08-17 13:22:28 CST. The ledger has exactly 230 rows and 230 unique task IDs, all with return code zero and no timeout. Both finalizers returned zero and the tmux controller exited. The reconciled outputs contain 200 real rows and 100 expanded PDE strategy rows; every primary `reference_nrmse` is finite. Accounted task-slot time is 10.43 hours.

Optimizer-update fairness was verified from raw payload counters. On each real dataset, `joint_mse_step1`, `joint_continuity_step1`, and `no_q_mlp` have identical backward counts: 9,000 on NASA battery, 54,000 on Starry Seebeck, 45,000 on Starry electrical, and 51,000 on Starry thermal. All PDE latent and embedded neural baselines have 128,000 backward passes.

Durable paired interpretation:

- NASA battery is the clear positive real case: continuity q has median NRMSE 0.2279 versus kNN 0.3304 and wins 10/10 paired seeds; MSE q wins 8/10.
- Starry Seebeck continuity reaches median 0.01263 and reduces catastrophic seeds relative to MSE, but kNN is 0.002192 and wins 10/10. Starry electrical and thermal have `NRMSE > 10` for every seed of every neural method, while kNN remains finite and strong.
- PDE q=16 continuity (0.2573) is slightly better than q=16 MSE (0.2651), wins 7/10 paired seeds, and has higher continuity AUC (0.7759 versus 0.7236). Both beat pooled no-q MLP and support mean in 10/10 seeds, but both lose to kNN-4 (0.2113) in 10/10.
- Longer training is not a universal repair. NASA does not improve consistently; Seebeck continuity improves but retains one catastrophic seed; PDE q=16 MSE loses to its shorter 300-epoch/two-step anchor in every seed. Therefore support-kNN's advantage is not explained by unequal epoch counts.
- Do not claim continuity loss universally improves real-data geometry or prediction. Its clean joint benefit is specific to the tested PDE block; prediction, geometry, and numerical stability remain separate endpoints.

Live source-of-truth files:

| Purpose | Path |
|---|---|
| Atomic status snapshot | `runs/matched_1000ep_real_pde_20260817/campaign_status.json` |
| Frozen manifest | `runs/matched_1000ep_real_pde_20260817/campaign_manifest.json` |
| Frozen task reservoir | `runs/matched_1000ep_real_pde_20260817/planned_tasks.jsonl` |
| Append-only terminal ledger | `runs/matched_1000ep_real_pde_20260817/task_status.jsonl` |
| Per-task logs | `runs/matched_1000ep_real_pde_20260817/logs/` |
| Atomic real results | `runs/matched_1000ep_real_pde_20260817/real/**/result.json` |
| Atomic PDE results | `runs/matched_1000ep_real_pde_20260817/pdebench/**/result.json` |
| Terminal analysis report | `runs/matched_1000ep_real_pde_20260817/analysis/MATCHED_1000EP_REAL_PDE_RESULTS.md` |
| Terminal audit | `runs/matched_1000ep_real_pde_20260817/analysis/terminal_audit.json` |
| Per-dataset and paired CSVs | `runs/matched_1000ep_real_pde_20260817/analysis/*.csv` |

Do not restart this completed controller. Treat the terminal audit and raw `result.json` files as authoritative; if the report is edited later, preserve every dataset-level failure and re-run the frozen analyzer before changing a numeric claim.

## 21. Completed PDEBench functional-baseline campaign

This campaign closes two explicit ICLR-readiness gaps without changing the promoted q method: a classical linear functional latent space and an information-matched learned operator.

| Field | Value |
|---|---|
| Frozen plan | `PDEBENCH_FUNCTIONAL_BASELINES_PLAN_20260822.md` |
| tmux session | `lvs_pde_functional_20260822` |
| campaign root | `runs/pdebench_functional_baselines_20260822/` |
| methods | `fpca_ridge`, `masked_deeponet` |
| seeds | 0--9 |
| total | 20 atomic jobs |
| GPUs | physical 4 and 5 only; dispatch threshold below 128 MiB |
| shared protocol | frozen PDEBench 64/16/32 trajectory split, 16 x 32 grid, random support ratio 0.3, exact existing seed/label query splits |
| DeepONet budget | maximum 128,000 optimizer/backward updates; validation checkpoints at 16k/32k/64k/128k |
| timeout | no campaign deadline; 360 minutes per DeepONet job |

The FPCA basis is fit only on complete train trajectories. Component count (2/4/8/16/32) and ridge penalty (1e-6/1e-4/1e-2/1) are selected on validation trajectories; test coefficients use support targets only. The DeepONet branch sees a 512-location support mask plus masked standardized support values, while the trunk sees query `(x,t)` coordinates. It never receives test query targets.

Terminal state verified at 2026-08-22 00:50 CST: `campaign_status.json` is `completed_all`; all 20 planned task IDs have one successful ledger row, with zero failures and zero timeouts. Automatic summarization returned zero and the tmux controller exited. Every atomic result is finite and contains exactly 11,488 query rows aligned to the same-seed support-kNN artifact. Query-target perturbation changes predictions by exactly zero. Accounted task time is 0.731 hours.

| Method | Median NRMSE | p90 | Max | Median label-p95 |
|---|---:|---:|---:|---:|
| support kNN-4 | **0.211295** | 0.218136 | 0.219243 | — |
| q=16 continuity | 0.257289 | 0.268806 | 0.270941 | — |
| FPCA + ridge | 0.275280 | 0.286818 | 0.291442 | 0.450618 |
| masked DeepONet | 0.313649 | 0.318488 | 0.320887 | 0.507529 |

FPCA selected 32 components and ridge 0.01 in all ten seeds. DeepONet executed all 128,000 updates per seed, but validation selected the 16k checkpoint in eight seeds and the 32k checkpoint in two. Its 345,217 trainable parameters exceed the q=16 continuity model plus train embeddings (11,777) by about 29 times, so insufficient nominal capacity or update count does not explain its gap. In strict paired tests, q=16 continuity beats FPCA and masked DeepONet in 10/10 seeds; support-kNN beats all three in 10/10. The bounded claim is therefore that q is competitive with the declared classical and neural functional baselines, not that it is the best support-aware PDE method.

Source-of-truth files:

| Purpose | Path |
|---|---|
| Live atomic status | `runs/pdebench_functional_baselines_20260822/campaign_status.json` |
| Frozen manifest | `runs/pdebench_functional_baselines_20260822/campaign_manifest.json` |
| Frozen task reservoir | `runs/pdebench_functional_baselines_20260822/planned_tasks.jsonl` |
| Append-only ledger | `runs/pdebench_functional_baselines_20260822/task_status.jsonl` |
| Per-job logs | `runs/pdebench_functional_baselines_20260822/logs/` |
| Raw results | `runs/pdebench_functional_baselines_20260822/{fpca_ridge,masked_deeponet}/seed*/result.json` |
| Terminal report | `runs/pdebench_functional_baselines_20260822/PDEBENCH_FUNCTIONAL_BASELINE_RESULTS.md` |
| Terminal audit | `runs/pdebench_functional_baselines_20260822/terminal_audit.json` |
| Consolidated rows and paired effects | `runs/pdebench_functional_baselines_20260822/{all_runs,method_summary,paired_effects}.csv` |

Do not restart this completed controller or write additional cells into this root. Treat the terminal audit and raw `result.json` files as authoritative. At the post-run host snapshot, physical GPUs 4--5 were empty and the other six cards were occupied by unrelated users.

## 22. Completed q–kNN reliability-selector confirmation

This campaign tests the highest-priority remaining method hypothesis: use only support-internal validation to route each unseen entity between the complementary global latent-q and local support-kNN predictors.

| Field | Value |
|---|---|
| Frozen plan | `Q_KNN_RELIABILITY_SELECTOR_PLAN_20260822.md` |
| tmux session | `lvs_q_knn_selector_20260822` |
| campaign root | `runs/q_knn_reliability_selector_confirm_20260822/` |
| datasets | NASA battery; Starry Seebeck; Starry electrical; Starry thermal |
| seeds | 20--29, separate from the earlier 0--9 matched-update table |
| total | 40 atomic jobs |
| latent component | q=8 `joint_continuity_step1`, 1,000 epochs, adaptive K4 calibration |
| local component | per-entity distance-weighted support kNN, up to five neighbors |
| selector | 75%/25% split inside external support; lower physical-unit validation MAE wins; ties go to kNN |
| GPUs | physical 4 and 5, below-128-MiB dispatch guard |
| timeout | no campaign deadline; 240 minutes per job |

The formal advancement gates were written before seed-20--29 query results: NASA within 5% of the better fixed component; every Starry dataset has zero catastrophic selector runs and median within 10% of kNN; pooled selector no worse than the better fixed component; every integrity/leakage gate passes. The non-deployable query-oracle component choice is saved only to quantify routing regret and must not be presented as a method.

Terminal observation: all 40/40 task IDs finished successfully, with zero failures, zero timeouts, zero leakage violations, successful automatic summarization, and an exited controller. The predeclared overall advancement gate failed. NASA selected q for only about one third of entities and the selector median NRMSE was 0.319538 versus 0.2450 for latent q. Starry Seebeck was stable but its selector median 0.002577 was about 15% worse than kNN's 0.002231. Electrical and thermal both passed: the selector matched kNN (0.497297 and 0.056145) and eliminated all latent-q catastrophes. The support-internal/query component-error correlation was only 0.166 on NASA but 0.948--0.998 on Starry, explaining why per-entity routing did not transfer. Do not promote this selector.

Source-of-truth files:

| Purpose | Path |
|---|---|
| Live status | `runs/q_knn_reliability_selector_confirm_20260822/campaign_status.json` |
| Frozen manifest | `runs/q_knn_reliability_selector_confirm_20260822/campaign_manifest.json` |
| Frozen task reservoir | `runs/q_knn_reliability_selector_confirm_20260822/planned_tasks.jsonl` |
| Append-only ledger | `runs/q_knn_reliability_selector_confirm_20260822/task_status.jsonl` |
| Per-job logs | `runs/q_knn_reliability_selector_confirm_20260822/logs/` |
| Atomic results | `runs/q_knn_reliability_selector_confirm_20260822/results/*/seed*/result.json` |
| Terminal report after completion | `runs/q_knn_reliability_selector_confirm_20260822/Q_KNN_SELECTOR_RESULTS.md` |

Do not launch another controller against this root. Refresh the live JSON, ledger, tmux session, and host GPU ownership before intervention. Query-oracle rows are diagnostic only; all deployable claims must use the support-internal selector row.

## 23. Completed train-entity-validation q–kNN regime gate

This follow-up tests a dataset-level policy selected without test entities: train q on 75% of prepared training labels, compare q and support-kNN over three support/query episodes on the held-out 25%, freeze the lower-median component, retrain q on all training labels, then evaluate unseen test entities. It also recomputes the failed entity-level selector as a paired anchor.

| Field | Value |
|---|---|
| Frozen plan | `HIERARCHICAL_Q_KNN_GATE_PLAN_20260822.md` |
| runner | `scripts/run_hierarchical_q_knn_gate_20260822.py` |
| tmux session | `lvs_hierarchical_q_knn_gate_20260823` |
| campaign root | `runs/hierarchical_q_knn_gate_confirm_20260822/` |
| datasets / seeds | NASA battery and three Starry properties; seeds 30--39 |
| total | 40 atomic jobs |
| intended GPUs | physical 0,1,6,7, dispatch only below 128 MiB |
| timeout | no campaign deadline; 240 minutes per job |

Static compile, 40-cell dry-run, frozen config assertions, and a 5-epoch NASA GPU smoke passed. The smoke verified disjoint meta-fit/meta-validation labels, five finite method outputs, positive optimizer counters, and query-target perturbation difference exactly zero.

Timestamped launch observation at 2026-08-23 12:19 CST: all eight GPUs became occupied by VLLM workers immediately after the smoke, so the controller waited with 40 pending and dispatched only after the selected cards became empty.

Advancement requires NASA to select q in at least 8/10 seeds and stay within 5% of q, every Starry dataset to select kNN in at least 8/10 seeds with zero catastrophes and within 5% of kNN, pooled performance no worse than the entity selector, and every integrity gate to pass. Failure ends this selector line without post-hoc threshold tuning.

Terminal observation reconciled 2026-08-24 15:17 CST: all 40/40 unique task IDs finished with return code zero, zero timeouts, 40 atomic results, 200 method rows, disjoint meta-fit/meta-validation entity sets, and maximum query-target perturbation difference exactly zero. The launcher initially recorded `completed_with_failures` only because report rendering called the unavailable optional `tabulate` package. The analyzer was minimally changed to emit Markdown directly, then reran successfully; `campaign_status.json` now records `completed_all` and summarize return code zero.

The predeclared advancement decision is **FAIL**. Training-entity validation selected q in exactly 8/10 NASA seeds and kNN in 10/10 seeds for each Starry dataset. All Starry gates passed: hierarchical prediction equals kNN, with median NRMSE 0.002254 (Seebeck), 0.519762 (electrical), and 0.054370 (thermal), and zero catastrophic runs. NASA failed the effect gate: hierarchical median NRMSE is 0.288962 versus 0.237327 for full q, a 21.76% increase rather than the allowed 5%. Although the gate chose q in 8/10 seeds, it matched the test-query component winner in only 6/10: seeds 32 and 33 incorrectly chose kNN, while seeds 36 and 39 incorrectly chose q. The two harmful kNN choices replaced strong q runs and shifted the median. Pooled hierarchical performance was no worse than the entity-selector anchor, but that does not rescue the failed NASA gate.

Durable conclusion: train-entity validation reliably detects the catastrophic Starry regime, but is not a sufficiently stable selector for NASA under the frozen split and three-episode score. Do not promote this hierarchical selector and do not tune a threshold on seeds 30--39. The selector line ends unless a genuinely new, independently testable hypothesis and fresh data split are predeclared.

Source-of-truth terminal artifacts:

| Purpose | Path |
|---|---|
| Status | `runs/hierarchical_q_knn_gate_confirm_20260822/campaign_status.json` |
| Raw results | `runs/hierarchical_q_knn_gate_confirm_20260822/results/*/seed*/result.json` |
| Terminal audit | `runs/hierarchical_q_knn_gate_confirm_20260822/terminal_audit.json` |
| Terminal report | `runs/hierarchical_q_knn_gate_confirm_20260822/HIERARCHICAL_Q_KNN_GATE_RESULTS.md` |
| Consolidated tables | `runs/hierarchical_q_knn_gate_confirm_20260822/{all_runs,method_summary,paired_effects}.csv` |

## 24. Active support-envelope projected-q confirmation

ICLR-facing hypothesis: retain the same explicit-q representation and decoder, but project each unseen entity's raw query predictions into a support-only envelope `[support_min - train_target_std, support_max + train_target_std]`. This tests failure control rather than another q/kNN router. It does not claim uncertainty calibration or superiority to kNN.

The multiplier 1 was selected once on completed development seeds 0--9 from the grid `{0, 0.25, 0.5, 1, 2}`. On that development block it retained NASA within 2.65% of raw q and reduced every Starry run below the catastrophic threshold. Formal seeds 40--49 are fresh relative to the 0--9, 20--29, and 30--39 confirmations; no threshold may be tuned on them.

| Field | Value |
|---|---|
| Frozen plan | `SUPPORT_ENVELOPE_PROJECTED_Q_PLAN_20260824.md` |
| runner | `scripts/run_support_envelope_projected_q_20260824.py` |
| tmux session | `lvs_support_envelope_q_20260824` |
| campaign root | `runs/support_envelope_projected_q_confirm_20260824/` |
| datasets / seeds | NASA battery and three Starry properties; seeds 40--49 |
| total | 40 atomic jobs |
| GPUs | physical 0,1,6,7 only, below-128-MiB dispatch guard |
| timeout | no campaign deadline; 240 minutes per job |

Static compile, 40-cell dry-run, exact config assertions, and a 5-epoch NASA GPU smoke passed. The smoke had finite projected metrics, zero envelope violations, query-leakage upper bound zero, and preserved the q representation artifact.

Timestamped initial observation at 2026-08-24 15:33 CST: controller alive, four seed-40 jobs dispatched only to physical GPUs 0,1,6,7, with 0 completed, 0 failed, 36 pending, and four running. GPUs 2--5 remained occupied by unrelated work and were not in the campaign GPU list.

Timestamped terminal observation at 2026-08-25: unrelated VLLM engines started on physical GPUs 0,6,7 after the campaign began. Seven cells failed with `CUDA driver initialization failed`; 33/40 succeeded, pending/running reached zero, and automatic summarization did not run. This is an incomplete infrastructure-affected campaign, not a confirmatory result. Preserve the 7 failures and do not promote the 33 survivors. A deliberate retry on a new root or an explicitly documented amendment is required to complete the frozen matrix.

Advancement requires all integrity gates, NASA projected-q median within 5% of raw q with zero catastrophes, each Starry projected median no worse than raw q with zero catastrophes, and pooled projected median below pooled raw q. Passing supports only failure-controlled explicit q; it does not require or imply beating kNN.

## 25. Completed real-data raw-q symbolic-interface development

This experiment directly tests the paper's downstream-utility claim on real data. It asks whether q calibrated from an entity's support set lets a compact symbolic surrogate fitted on one half of unseen test entities generalize to the other half. With no known real generating law or true q, it tests cross-entity symbolic transfer, not physical-law recovery.

| Field | Value |
|---|---|
| Frozen plan | `REAL_SYMBOLIC_Q_INTERFACE_PLAN_20260824.md` |
| runner | `scripts/run_real_symbolic_q_interface_20260824.py` |
| q source | completed matched-update real q=8 continuity artifacts, seeds 0--9 |
| development | seeds 0,1,2; 4 datasets × 4 regimes = 48 PySR cells |
| reserved confirmation | seeds 3--9; 112 cells, not yet exposed to symbolic fitting |
| regimes | condition+q, condition only, condition+support target statistics, full observed x |
| symbolic split | deterministic 50/50 split of test entities into symbolic-fit and symbolic-held-out |
| PySR budget | 60 iterations, maxsize 24, same operators, at most 1,200 fit rows |
| tmux | `lvs_real_symbolic_dev_20260824` |
| root | `runs/real_symbolic_q_interface_20260824/` |

All regimes standardize inputs and targets using symbolic-fit entities only and remove fit-constant columns. Starry condition-only uses temperature; NASA retains its five within-entity varying measurement/operation features. q and support statistics use external support targets only; symbolic held-out query targets are scoring-only.

Static compile, 48-cell dry-run, dataset/split audit, and a 2-iteration NASA PySR smoke passed under the historical PySR 1.5.10 environment. The GPU environment lacks PySR and was not modified. Initial resource snapshot at 2026-08-24 15:49 CST: two PySR workers consumed approximately 11 user CPU cores; host load was 44/128, within the frozen maximum.

Terminal result: 48/48 PySR cells succeeded, with finite predictions and intact entity-level boundaries, but the predeclared development gate **failed**. `condition+q` beat condition-only in only 2/12 paired cells, support statistics in 3/12, and full x in 3/12. Dataset median held-out NRMSE for raw q was 1.9267 (NASA), 0.005639 (Seebeck), 0.12968 (electrical), and 0.08876 (thermal); support statistics were better on all four medians except NASA condition-only was best there. Seeds 3--9 remain unused; do not launch the raw-q confirmation.

Every raw-q formula used at least one q coordinate, but the selected direction changed across seeds: NASA q8/q7/q6; electrical q5/q6/q5; thermal q2/q1/q5; Seebeck q6, q1/q3, q3/q5. Durable conclusion: q contains entity information, but arbitrary coordinate rotation/permutation makes raw q a poor symbolic vocabulary. Full results are in `runs/real_symbolic_q_interface_20260824/REAL_SYMBOLIC_Q_INTERFACE_DEVELOPMENT_RESULTS.md`.

## 26. Completed functional-canonical-q symbolic development

This is the only follow-up authorized by the raw-q formula clue. It learns a two-dimensional response-aligned linear rotation of q using symbolic-fit entities only, then reruns the same held-out-entity PySR score. It does not change the neural predictor and does not inspect formal seeds.

| Field | Value |
|---|---|
| Frozen plan | `FUNCTIONAL_CANONICAL_Q_SYMBOLIC_PLAN_20260825.md` |
| runner | `scripts/run_real_symbolic_q_interface_20260824.py --phase canonical_development` |
| cells | four datasets × seeds 0,1,2 = 12 |
| canonicalization | 64-dimensional fixed-RFF response signatures, seed 20260808; two-component PLS q rotation |
| information boundary | signatures and PLS use symbolic-fit entity query rows only; held-out q remains support-calibrated and held-out query targets are scoring-only |
| PySR budget | unchanged: 60 iterations, maxsize 24, same operators, at most 1,200 fit rows |
| tmux | `lvs_functional_canonical_q_dev_20260825` |

Static compile, 12-cell dry-run, per-dataset fit/held-out audit, and a two-iteration Seebeck canonical-q smoke passed. The smoke is structural only and has no scientific interpretation. The formal seeds 3--9 are still sealed.

Advancement requires 12 finite terminal cells; pooled canonical-q NRMSE below raw q, condition-only, and support statistics; at least 9/12 wins over raw q; dataset-level wins over condition-only and support statistics on at least three of four datasets; and no higher median formula complexity than raw q. Any failure ends this exact canonicalization without post-hoc dimension or signature tuning.

Terminal observation: 12/12 cells completed with finite outputs, but the advancement gate failed. Canonical q beat raw q in 6/12 cells rather than the required 9/12; pooled canonical NRMSE was 0.254636 versus raw q 0.120592, condition-only 0.094279, and support statistics 0.044671, while median complexity rose from 8.0 to 10.5. Formal seeds 3--9 remain sealed. This terminal result is now further limited by the dataset/protocol audit in section 27: all three Starry sources inherit an invalid prepared dataset, and NASA used only four symbolic-fit entities. Preserve the procedural FAIL, but do not promote it to a general scientific claim that q lacks downstream symbolic value.

## 27. Negative-result and dataset-eligibility reassessment

On 2026-08-25 the user required a method-favorable diagnostic order: audit dataset eligibility, data construction, information flow, evaluation protocol, and optimization before interpreting a negative result as a method limitation. This is a working diagnostic priority, not permission to hide contrary evidence. Exclusion criteria must be independent of observed performance and must remove positive and negative cells from the same invalid construction together.

The full internal audit is [NEGATIVE_RESULT_REASSESSMENT_20260825.md](NEGATIVE_RESULT_REASSESSMENT_20260825.md). No new model training was launched. Durable findings:

- `data/application_full_features` Starry is invalid for the declared entity-level q task. Its actual label is composition, not sample ID; each selected composition combines many samples and papers, while 86 constant element-fraction features already expose the composition. Electrical also mixes raw/log/ln targets and inverse/powered temperature coordinates; thermal mixes total/lattice/electronic targets; Seebeck contains inverse-temperature and extreme records. All positive and negative claims derived from this prepared source are withdrawn pending strict reconstruction.
- Reviewer-clean Starry uses sample IDs, but fuzzy property filtering remains. ZT passed this audit; Seebeck has 2/80 affected entities and is provisional; electrical has 21/80 affected entities and thermal 41/80, so both are invalid pending reconstruction.
- A filesystem content audit found at least 1,035 `result.json` files explicitly linked to full-features Starry and 234 linked to reviewer-clean electrical/thermal. Another 148 reviewer-clean Seebeck files are provisional; 117 reviewer-clean ZT files remain eligible. Counts describe atomic artifacts, not independent scientific hypotheses.
- NASA battery is a strong in-scope real task and must not be excluded because of symbolic failure, but the original prepared NASA source is now also invalid: duplicate battery files crossed the entity split and same-cycle response diagnostics were used as query features. Historical NASA gates retain their procedural records but are not eligible paper evidence until the clean protocol is rerun.
- MATR battery remains in scope; random support/query mostly tests interpolation, so kNN's advantage does not answer the forward/extrapolative task. UCI gas has only 10 entities and about two test entities and is a small-sample stress test. The engine dataset previously removed by the user remains outside manuscript scope. PDEBench is an external high-dimensional trajectory/interpolation stress test: its q-over-FPCA/DeepONet conclusions stand, while kNN-over-q is protocol-local rather than a core-method refutation.
- The all-46 recovery denominator remains intact, but 28/46 is a strict symbolic-recovery-protocol pass rate, not the fraction of tasks where q is useful. Three negative outcomes are numerical overflow and three are weak-control/identifiability cases. Several of the 12 remaining symbolic non-recoveries have excellent q prediction NRMSE, so they are downstream-readout failures rather than latent prediction failures.
- The beginner-readable main report now carries a prominent `PARTIALLY SUPERSEDED` warning. Do not quote its pooled real-data or Starry conclusions until the tables are rebuilt from eligible datasets only.

## 28. Mandatory real-data q-to-symbolic structure-refinement loop

The user has made one paper objective non-negotiable: at least one eligible real dataset must demonstrate a complete `latent q -> symbolic regression -> structure modification -> relearn q -> more interpretable symbolic expression` loop. A single failed development gate must trigger diagnosis and iteration, not cancellation of this objective. The expression need not be the unique true physical law, but it must be a held-out-entity-validated, stage-wise interpretable surrogate that materially guides model structure.

The frozen milestone is [REAL_Q_SYMBOLIC_STRUCTURE_LOOP_MILESTONE_20260825.md](REAL_Q_SYMBOLIC_STRUCTURE_LOOP_MILESTONE_20260825.md). NASA battery remains the primary real system because its scientific task matches early-support-to-later-query system identification, but the historical 10/10 q-over-kNN result has been withdrawn as evidence after the duplicate-ID and same-cycle-feature audit. MATR battery is the second system. Invalid Starry data cannot be used to manufacture success.

The next implementation uses the reviewer-clean NASA 13/5 outer split and its three frozen 8/5 inner splits described in section 29. The primary protocol is blocked early-cycle support to later-cycle query; random support remains secondary. Do not expose arbitrary raw q coordinates as the only symbolic vocabulary: derive response-functional coordinates from q plus the frozen decoder, identify motifs stable across inner splits/seeds, implement the smallest formula-informed backbone with a measured residual, relearn q, and rerun the symbolic comparison.

Completion requires intact information boundaries, a final formula that uses both a q-derived functional coordinate and a physical condition, improvement over condition-only symbolic regression on held-out entities, a motif recurring across most inner splits/seeds, structured prediction within 5% of the original q median NRMSE, and improvement in at least two of held-out symbolic error, formula complexity, and motif stability after the structure/refit loop. Until those conditions hold, report progress and blockers but do not mark this research objective complete or infer that the core method lacks real downstream value.

## 29. Reviewer-clean NASA protocol and blocked-q implementation (2026-08-25 15:43 CST)

This transition supersedes the NASA entity/split details in section 28; the closed-loop objective itself is unchanged.

Data qualification found three independent flaws in the original prepared NASA source:

- B0025--B0028 occur in two raw batch directories as exact row-for-row duplicate files. Because the old label included the directory name, B0025, B0026, and B0028 crossed the old train/test split under different labels.
- The old query inputs included `voltage_min`, `temperature_mean`, and `current_abs_mean`, all computed from the same discharge cycle whose capacity is the target. They are not clean forward-prediction conditions.
- The old corpus mixed later batches whose own README warns that several very-low-capacity runs have not been explained; 19 exact zero-capacity rows occurred across 12 old labels.

Consequently, all historical NASA values—including the 1,000-epoch 10/10 q-over-kNN comparison—are downgraded to non-paper evidence. This is a data/protocol invalidation of both positive and negative results, not a negative judgment on q.

The new reproducible preparer is `scripts/prepare_nasa_battery_reviewer_clean_20260825.py`. It produced a frozen development cohort under `data/real_datasets2/prepared/nasa_battery_reviewer_clean_20260825/` with these properties:

- 18 unique battery IDs from B0005--B0040 whose source README does not carry the unexplained-low-capacity warning;
- exact verification and removal of the four duplicated battery files;
- five excluded cycles from B0033 based on a result-independent physical rule: measured discharge-current q90 below 0.5 A; all retained capacities are finite and positive;
- only forward-available inputs: discharge index, ambient temperature, nominal load-current amplitude in `{1,2,4}` A, and documented cutoff voltage;
- one frozen outer battery per documented protocol family: 13 train batteries / 5 outer-test batteries, 1,191 / 444 rows, zero identity overlap;
- three frozen inner splits, each 8 meta-fit / 5 structure-validation batteries with one validation battery per protocol family. Their exact labels are recorded in `qualification_audit.json`.

`scripts/run_iclr_real_discovery.py` now accepts explicit `--support-split-mode prefix --support-order-column discharge_index`, passes the same prefix rule into q calibration, and saves both `train_label_q.csv` and `training_checkpoint.pt`. A two-epoch CPU smoke completed at `runs/_tmp_archive_20260828/lvs_nasa_clean_smoke` (migrated from `/tmp` on 2026-08-28): 13 train q rows, 5 test q rows, a loadable checkpoint, 132 support rows, and 312 query rows. For every outer battery the last support cycle is strictly before the first query cycle. The smoke metrics are structural only and must never be quoted scientifically.

Resource observation at takeover: no relevant tmux session or project training process was active. Both `nvidia-smi` and `nvidia-smi pmon` returned `couldn't communicate with the NVIDIA driver`; therefore GPU availability is unknown, not empty. No formal clean-NASA training has been launched. Before launch, recheck the driver/device boundary, freeze the exact command and stopping rule, and preserve unrelated jobs.

Timestamped launch authorization and resource refresh at 2026-08-25 16:04 CST: the user explicitly confirmed the frozen 35-cell command covering five methods, q dimensions 2/4 where applicable, and seeds 0--4 at 1,000 epochs, with prefix support and no row caps. Host `nvidia-smi` showed physical GPUs 2, 3, and 4 at 0 MiB and GPU 5 at 4 MiB, all at 0% utilization and with no compute processes; GPUs 0, 1, 6, and 7 were occupied by unrelated jobs. The 128 MiB value is only the pre-dispatch empty-card threshold, not a training-memory limit. No relevant controller or result file existed under `runs/nasa_battery_reviewer_clean_anchor_20260825/` immediately before launch.

## 30. Completed reviewer-clean NASA outer anchor and functional-coordinate audit

Terminal observation at 2026-08-25 16:09:46 CST: the exact user-confirmed outer-anchor command completed with 35/35 unique atomic results, 35/35 zero launcher return codes, and zero non-finite `reference_nrmse` values. The 20 latent cells all contain `train_label_q.csv`, `test_label_q.csv`, `training_checkpoint.pt`, query predictions, continuity curves, nearest-neighbor tables, and geometry figures. Every cell records 1,191 train rows, 444 outer-test rows, 13/5 disjoint batteries, 132 prefix-support rows, 312 later-cycle query rows, 1,000 epochs, and the frozen four-start support-only calibration settings.

Five-seed outer development medians are: support kNN 0.593400 reference NRMSE; q=4 MSE 1.085538; q=4 continuity 1.097999; q=2 MSE 1.342163; q=2 continuity 1.876252; Random Forest 1.801450; and no-q MLP 1.887869. Both q=4 methods beat no-q and Random Forest in 5/5 paired seeds, but lose to prefix-support kNN in 5/5. The paired median relative gap versus kNN is +82.94% for q=4 MSE and +85.04% for q=4 continuity. q=4 beats its q=2 counterpart in 4/5 seeds for both losses. Continuity q=4 beats MSE q=4 in only 1/5 seeds, so the continuity loss is not a clean prediction improvement in this blocked clean protocol.

The representation audit gives a different and useful result. Across all ten seed pairs, train-q distance geometry has median Spearman 0.996333 for continuity q=4 and 0.988695 for continuity q=2, versus 0.287080 and 0.191770 for MSE q=4/q=2. Continuity q=4 named decoder-response coordinates have cross-seed median rank correlations 0.799 for cycle-1 capacity, 0.692 for cycle-28 capacity, 0.692 for early fade, 0.662 for mid fade, and 0.275 for fade acceleration. On the 13 training batteries, decoder cycle-1 capacity correlates with the independently computed empirical cycle-1 descriptor at median Spearman 0.934 across continuity q=4 seeds. These are development clues from a small training cohort, not held-out physical validation.

Durable interpretation: after removing duplicate identities, same-cycle leakage features, and random support interpolation, explicit q still provides entity information beyond support-blind MLP/RF, but it is not the best pure predictor; a simple early-prefix kNN/extrapolation anchor is substantially stronger. The continuity objective buys exceptionally stable cross-seed q geometry and named functional coordinates, not lower prediction error or uniformly better within-run geometry. This prediction--representation trade-off motivates the inner-split symbolic motif stage rather than invalidating q. Outer results may not be used to fit a formula; structure selection must use the three frozen inner splits, and the final outer loop remains development evidence because the cohort has been exposed.

Source-of-truth artifacts:

| Purpose | Path |
|---|---|
| Frozen manifest | `runs/nasa_battery_reviewer_clean_anchor_20260825/experiment_manifest.json` |
| Append-only terminal ledger | `runs/nasa_battery_reviewer_clean_anchor_20260825/launcher_status.jsonl` |
| Atomic results | `runs/nasa_battery_reviewer_clean_anchor_20260825/nasa_battery_capacity_reviewer_clean/**/result.json` |
| Consolidated rows | `runs/nasa_battery_reviewer_clean_anchor_20260825/all_runs.csv` |
| Method summary | `runs/nasa_battery_reviewer_clean_anchor_20260825/method_summary.csv` |
| Functional-coordinate outputs | `runs/nasa_battery_reviewer_clean_anchor_20260825/functional_coordinate_analysis/` |

## 31. Authorized inner-split q matrix

Timestamped authorization and resource snapshot at 2026-08-25 16:20 CST: the user explicitly confirmed the displayed 30-cell inner-split command. It covers the three frozen 8/5 meta-fit/structure-validation splits, `joint_continuity_step1` and `joint_mse_step1`, q=4, seeds 0--4, 1,000 epochs, and the same prefix-support calibration settings as the outer anchor. The output root is `runs/nasa_battery_reviewer_clean_inner_q_20260825/`. Immediately before launch, host `nvidia-smi` showed GPUs 2, 3, and 4 at 0 MiB and GPU 5 at 4 MiB with no compute processes; the output root did not exist and no relevant tmux controller was present. The success gate is 30/30 zero-return-code cells with finite metrics and complete q/checkpoint artifacts. No automatic retry is authorized after a terminal failure.

## 32. Completed inner-split q matrix and functional-coordinate audit

Terminal observation at 2026-08-25 16:37 CST: the authorized controller completed 30/30 unique cells, all 30 launcher return codes are zero, and every `reference_nrmse` is finite. All 30 cells contain a checkpoint, train/test q tables, and predictions. The subsequent frozen-decoder analysis completed 30/30 cells and produced exactly 30 copies of each required artifact: `metadata.json`, `decoder_probes.csv`, `functional_coordinates.csv`, `train_coordinate_descriptors.csv`, and `train_descriptor_correlations.csv`. Every metadata record says `outer_targets_used: false`.

Structure-validation prediction does not favor the continuity loss. Per-split five-seed median reference NRMSE is 1.386079 versus 0.917472 for continuity/MSE on inner0, 1.378179 versus 0.908014 on inner1, and 1.169410 versus 1.072549 on inner2. Continuity wins only 4/15 paired split/seed cells; its paired median delta is +0.342803, or +32.86%. Do not describe continuity as a NASA prediction improvement.

The representation result replicates strongly across all three inner cohorts. Within-split q-distance cross-seed Spearman medians are 0.996716, 0.999453, and 0.998905 for continuity, versus 0.485769, 0.308429, and 0.390805 for MSE. The decoder-derived cycle-1-capacity and early-fade coordinates have median-of-split cross-seed stability 0.904762 and 0.821429; continuity exceeds MSE for both in 3/3 splits. Their matched empirical-descriptor correlations have median-of-split medians 0.880952 and 0.523810, respectively. Mid fade has a weak split floor (0.071429 matched empirical correlation), and acceleration has weak overall empirical alignment (0.119048), so neither should guide the first structural modification.

Durable interpretation: continuity q is the primary symbolic-interface source because it supplies reproducible response geometry and two named functional coordinates, not because it minimizes NRMSE. The next frozen Stage C comparison must use all three inner splits and an equal PySR budget for condition-only, condition+support summaries, condition+raw q, and condition+functional q. The primary functional vocabulary is cycle-1 capacity plus early fade; discharge index is the required physical condition, while ambient temperature, load current, and cutoff voltage remain eligible when variable. No symbolic formula, motif, or decoder backbone has yet been selected, and no new symbolic command has been authorized.

Source-of-truth artifacts:

| Purpose | Path |
|---|---|
| Training ledger | `runs/nasa_battery_reviewer_clean_inner_q_20260825/launcher_status.jsonl` |
| Atomic results | `runs/nasa_battery_reviewer_clean_inner_q_20260825/nasa_battery_capacity_reviewer_clean_inner*/**/result.json` |
| Consolidated prediction rows | `runs/nasa_battery_reviewer_clean_inner_q_20260825/all_runs.csv` |
| Functional audit report | `runs/nasa_battery_reviewer_clean_inner_q_20260825/INNER_Q_FUNCTIONAL_RESULTS.md` |
| Per-cell and aggregate functional outputs | `runs/nasa_battery_reviewer_clean_inner_q_20260825/functional_coordinate_analysis/` |
| Reproducible stability aggregator | `scripts/aggregate_nasa_inner_functional_coordinates_20260825.py` |

## 33. Completed reviewer-clean NASA Stage C symbolic-structure screen and training-dynamics diagnosis

The exact frozen command in `NASA_INNER_SYMBOLIC_STRUCTURE_PLAN_20260825.md` was authorized by the user and completed on 2026-08-25 with process exit code zero. It ran 90 PySR cells: three inner splits × five seeds × condition-only, condition+prefix-support summaries, and raw/functional q interfaces for both `joint_continuity_step1` and `joint_mse_step1`. Every formula used the same 60-iteration/maxsize-24/operator budget. No cell was retried or removed. Julia printed several nonfatal `Distributed.ProcessExitedException` cleanup messages after workers exited, but the affected cells returned successful finite results and the controller completed normally; preserve this as a runtime anomaly, not a scientific failure.

The post-run audit is terminal and complete: 90 `result.json`, 90 `predictions.csv`, 90 Pareto fronts, 90 input scalers, 90 status lines, 90 unique expected cells, all finite metrics/predictions, exact 8 meta-fit / 5 structure-validation entity isolation, strict prefix ordering, and a 30-source query-target perturbation maximum feature difference of exactly zero. The root status is `completed_all`.

Frozen gate decision: **3/5 PASS; overall FAIL**. Integrity passed. Downstream value failed: continuity functional-q validation NRMSE median is 1.755191 versus 0.936510 for condition-only and 1.033182 for support statistics, with only 4/15 paired wins against each rather than the required 9/15. Motif recurrence passed: a selected expression uses `discharge_index` plus at least one frozen functional coordinate in 12/15 continuity cells, split counts 3/5, 5/5, and 4/5. Readability failed: continuity functional-q median complexity 13 exceeds raw-q 11. The representation diagnostic passed narrowly because the same broad motif occurs 12/15 for continuity versus 11/15 for MSE.

The strongest diagnostic is interface shift, not leakage. Physical conditions reach at most about 2.24 meta-fit standard deviations on structure validation. Raw q reaches group-median maximum absolute z-scores of 22.19 for continuity and 12.38 for MSE, with a maximum 35.06. Unbounded PySR expressions then place out-of-domain q in exponentials or near-zero denominators; the largest finite NRMSE is `6.356848e44`. Functionalization reduces the shift and improves continuity functional versus raw in 10/15 paired cells (median NRMSE 1.755 versus 3.081), but its validation maximum-|z| medians remain 7.37/4.42 for continuity/MSE and three functional cells exceed NRMSE 10.

Durable interpretation: the wide `cycle + q-functional` motif is a reproducible structural clue, but the current full-curve train embedding versus prefix-support calibrated test-q interface is not prediction-valid or safe enough to advance directly. The train q embeddings were learned jointly from complete meta-fit curves, whereas validation q was inferred with a frozen decoder from early support only. These are different information and optimization pathways. Current results reject the unbounded, distribution-unmatched symbolic interface; they do not show that q lacks entity information. Do not select one attractive Stage C equation or tune its thresholds/operators on these same 90 outcomes.

The next experiment must be independently frozen before execution. Highest-priority protocol repair: calibrate meta-fit-entity q from the same 30% prefix support used for validation q, then fit formulas on those support-inferred coordinates; audit support Jacobian singular values/conditioning and q-to-train-manifold distance; use bounded functional-coordinate interfaces without `exp(q)`, nested exponentials, or unprotected q denominators. Only if a new validation gate beats condition-only may the provisional `initial capacity / fade coefficient + cycle` motif enter a minimal structured decoder with a measured residual. The mandatory q → symbolic → structure → relearn-q → second-symbolic loop remains unfinished.

Source-of-truth artifacts:

| Purpose | Path |
|---|---|
| Frozen Stage C plan | `NASA_INNER_SYMBOLIC_STRUCTURE_PLAN_20260825.md` |
| Runner | `scripts/run_nasa_inner_symbolic_structure_20260825.py` |
| Reproducible analyzer | `scripts/analyze_nasa_inner_symbolic_structure_20260825.py` |
| Terminal status and manifest | `runs/nasa_battery_reviewer_clean_inner_symbolic_20260825/{status,manifest}.json` |
| Raw consolidated results | `runs/nasa_battery_reviewer_clean_inner_symbolic_20260825/results.csv` |
| Integrity and frozen gates | `runs/nasa_battery_reviewer_clean_inner_symbolic_20260825/{integrity_audit,gate_decision}.json` |
| Beginner-readable Stage C analysis | `runs/nasa_battery_reviewer_clean_inner_symbolic_20260825/STAGE_C_ANALYSIS.md` |
| Cell-level formulas, shifts, and tails | `runs/nasa_battery_reviewer_clean_inner_symbolic_20260825/cell_diagnostics.csv` |
| Project-specific theory note | `NEURAL_TRAINING_DYNAMICS_FOR_LATENT_Q_20260825.md` |

The theory note derives the exact q/first-layer affine gauge symmetry, q-only gradient competition between prediction and continuity, local calibration Hessian `J_support^T J_support`, and the predicted role of lazy/feature-learning and spectral dynamics. Its main testable implication is that stable continuity geometry can coexist with worse prediction when support calibration is ill-conditioned. It separates literature-backed idealized theory, exact project algebra, and project hypotheses; do not cite all theory claims as already experimentally verified.

## 34. Repository stage snapshot (2026-08-26)

The user authorized synchronizing the current research state to a new remote branch. Local branch `research/latent-q-stagec-20260826` was created from the existing research worktree without rebasing or discarding any pre-existing changes. The historical research base remains commit `2b13869`; the remote `main` head observed immediately before branch creation was newer (`31e71df`), so any later integration should merge or rebase deliberately rather than assuming this snapshot already contains current `main`.

The snapshot scope is source code, experiment/analyzer scripts, tests, the explicit handoff and research reports, and the compact frozen Stage C evidence chain. For Stage C this includes the 90-cell consolidated results, cell-level formula diagnostics, manifest/status, integrity audit, gate decision, paired/method/motif summaries, and beginner-readable analysis. Per-cell predictions, Pareto fronts, checkpoints, downloaded datasets, logs, local environments, and the broader generated `runs/` tree remain ignored. This boundary keeps the branch reviewable while preserving the numbers needed to audit the terminal Stage C claims.

Pre-commit verification on the snapshot worktree completed with `54 passed` under the active Python 3.11 GPU environment; the only warnings were the expected undefined R-squared values for fewer than two samples. The test fixture was minimally updated to carry the production-default `support_split_mode="random"` argument introduced by the real-data runner.

Remote synchronization completed at 2026-08-26 15:44 CST. Snapshot commit `0c3eab6` was pushed to `origin/research/latent-q-stagec-20260826` at `git@github.com:shiyan688/latent_variable_discovery.git`, and the local branch now tracks that remote branch.

## 35. Cross-machine portability normalization (2026-08-26)

After the remote snapshot, the user requested a portability audit and authorized remediation. The tracked core package had no user-home dependency, but 23 campaign/controller scripts selected a repository-local `.venv-lvs-gpu/bin/python`; the Stage C frozen command named the original host's PySR interpreter; the checked Stage C manifest stored absolute q/output roots; and README overstated that symbolic code/results were absent while understating the external artifact boundary.

The controllers now launch child tasks with `sys.executable`, so the active environment is inherited rather than guessed from a directory name. The shell PDE waiter accepts `PYTHON_BIN` and otherwise uses `python`. The Stage C runner writes repository-relative q/output roots with `path_base: repository_root`; the checked manifest was normalized in the same non-scientific path fields; and the analyzer accepts `--q-root` for a restored upstream artifact location. Frozen metrics, formulas, gates, splits, seeds, and predictions were not changed. The manifest retains its execution-time plan/runner hashes and adds hashes for the post-run portable revisions plus an explicit `scientific_outputs_changed: false` marker. PySR 1.5.10 was verified from the historical environment metadata and is declared as the `symbolic` optional dependency; `scipy` and `h5py` are declared under `experiments`.

README now distinguishes three levels: core/synthetic code that runs from a clean clone, compact Stage C summaries that can be inspected without raw artifacts, and full real/PDE/Stage C reruns that require external data or upstream q/checkpoint artifacts. A portability regression test scans only Git-tracked executables in a worktree (or all exported executables without `.git`) and rejects user-home or fixed historical-virtualenv bindings; a second test requires relative Stage C manifest paths. Full verification at this transition is 56 tests passed with only the two expected small-sample R-squared warnings. The staged candidate tree was exported without `.git` or ignored local files and passed the same 56 tests; its CLI listed the tracked expression library and completed a two-epoch CPU workflow smoke with all artifacts written under the supplied output root. Large datasets and omitted raw run artifacts remain external by design; do not claim that cloning this branch alone reproduces all paper experiments.

## 36. Completed support-matched q interface diagnostic (2026-08-26 16:38 CST)

The next independently frozen diagnostic is `NASA_SUPPORT_MATCHED_Q_DIAGNOSTIC_PLAN_20260826.md`. It reuses all 30 completed inner-q checkpoints without retraining the decoder. Each meta-fit entity is recalibrated from its earliest 30% target rows using a leave-one-entity-out train-q prior; structure-validation q uses the same prefix protocol and all eight meta-fit embeddings as its prior. It saves raw/functional shift, calibration dispersion, full-curve-to-prefix q displacement, q-manifold distance, and support-Jacobian singular values/condition/effective rank. Query-target perturbation must leave q unchanged.

The predeclared advancement gate requires 30/30 integrity and exact held-out reproduction, at least a 50% reduction of the continuity raw-q median max-|z| from 22.1915, continuity functional median max-|z| at most 3.0, and at least 12/15 continuity functional cells at most 6.0. Jacobian conditioning is diagnostic only. No symbolic Stage C2 or decoder modification is automatic.

A non-counted single-checkpoint GPU smoke passed: leakage difference 0, saved-q reproduction maximum `9.31e-10`, NRMSE difference `1.05e-7`, matched raw/functional max-|z| 6.328/2.217, Jacobian median smallest singular value 0.1111, median condition number 181.18, and median effective rank 4. The full 30-cell matrix was then launched at 16:31 CST after host `nvidia-smi` showed GPUs 0 and 7 at 0 MiB/0%; GPUs 1--6 were occupied by unrelated jobs.

Terminal reconciliation found 30/30 successful cells, exact 8/5 entity counts, maximum query-target leakage difference zero, maximum saved-q reproduction error `1.192e-7`, and maximum NRMSE reproduction difference `1.668e-7`. Integrity and reproduction gates passed. The frozen shift gates failed, so the diagnostic does **not** advance directly to Stage C2. For continuity q, support matching reduces median raw max-|z| from 22.1915 to 9.7086 and functional max-|z| from 7.3658 to 4.2823, but the functional median remains above 3.0 and only 11/15 cells are at most 6.0 rather than the required 12. MSE gives raw/functional medians 9.4246/4.8911 and 13/15 functional-safe cells.

The remaining failure is more consistent with off-manifold calibration than local rank collapse. Across all 30 cells, functional shift has Spearman 0.572 with nearest support-matched q-manifold distance (`p=0.000959`) but only 0.218 with support-Jacobian condition number (`p=0.247`). At the entity level the corresponding correlations are 0.467 (`p=1.75e-9`) and 0.031 (`p=0.711`). Jacobians retain median effective rank 4, although continuity condition numbers remain high (cell-median 137.13). The worst continuity cells are concentrated in B0039 on inner split 2 and B0033 on split 1, and they also have large nearest-manifold q distances. This motivates a convex-support parameterization rather than relaxed gates.

Source-of-truth artifacts:

| Purpose | Path |
|---|---|
| Frozen plan | `NASA_SUPPORT_MATCHED_Q_DIAGNOSTIC_PLAN_20260826.md` |
| Terminal status and gates | `runs/nasa_support_matched_q_diagnostic_20260826/{status,gate_decision}.json` |
| Cell summaries and raw q/Jacobians | `runs/nasa_support_matched_q_diagnostic_20260826/{all_cells,all_support_matched_q,all_support_jacobians}.csv` |
| Readable report | `runs/nasa_support_matched_q_diagnostic_20260826/DIAGNOSTIC_REPORT.md` |

## 37. Completed convex-support q diagnostic (2026-08-26 16:51 CST)

The independently frozen next experiment is `NASA_CONVEX_SUPPORT_Q_DIAGNOSTIC_PLAN_20260826.md`. It keeps all 30 decoders fixed and constrains each structure-validation q to `softmax(alpha) @ Q_anchor`, where the eight anchors are the same cell's support-matched meta-fit q values. Only the earliest 30% support targets optimize alpha; query targets remain scoring-only. This is the smallest direct test of the observed off-manifold failure and preserves an explicit four-dimensional q rather than switching methods.

Advancement requires 30/30 integrity, exact simplex and leakage checks, all 15 continuity raw-q shifts at most 3.0, the unchanged functional median/tail thresholds of 3.0 and 12/15 at most 6.0, and prediction retention: continuity median NRMSE within 5% of the unconstrained support-matched comparator with at least 10/15 per-cell ratios at most 1.10. No threshold will be lowered after outcomes are observed, and passing would authorize only a separately frozen bounded symbolic Stage C2.

Static compilation, CLI checks, `git diff --check`, and the full 56-test suite passed before the first cell. A non-counted continuity inner0/seed0 smoke then passed all structural checks: 8 anchors, 5 validation entities, zero query leakage, simplex error `7.03e-8`, positive weights, raw/functional max-|z| 2.192/2.307, and effective-anchor median 1.775. Its convex NRMSE is 1.9426 versus 1.6294 unconstrained (ratio 1.192), which is an unfavorable early predictive observation but is not a gate decision; the full frozen 15-cell continuity distribution must decide retention.

Terminal reconciliation found 30/30 successful cells and **3/4 frozen gates passed**. Integrity, simplex containment, and functional shift all pass. Continuity raw max-|z| has median/maximum 2.321/2.547; functional max-|z| is 2.351/2.542, so all 15/15 cells meet the functional tail threshold. MSE gives raw 1.824/2.426 and functional 1.664/2.125. Thus a support-only q can be kept in a safe coordinate regime without changing the decoder.

Prediction retention fails decisively. Continuity convex NRMSE has median 1.696 versus 1.193 for unconstrained support-matched q, the per-cell ratio median is 1.213, and only 6/15 cells stay within 10%; the frozen requirements were a median within 5% and at least 10/15 retained cells. MSE similarly worsens from 0.9175 to 1.197 with only 5/15 retained cells. Therefore this exact convex-support calibration does not advance to Stage C2.

Failure diagnosis identifies a specific optimization/parameterization confound rather than a generic rejection of bounded q. The near-one-hot best-anchor initialization is selected for 61/75 continuity entities and 56/75 MSE entities. Continuity entity-level effective-anchor count has median 1.0003 and cell-level median 1.0003; the method often degenerates into selecting one training battery. Performance is split-dependent: continuity retains 1/5, 5/5, and 0/5 cells on inner0/inner1/inner2. A next repair may test deterministic minimum-change coordinate bounding, but it must be frozen before inspecting its outcomes and treated as sequential development evidence.

Source-of-truth artifacts:

| Purpose | Path |
|---|---|
| Frozen plan | `NASA_CONVEX_SUPPORT_Q_DIAGNOSTIC_PLAN_20260826.md` |
| Terminal status and gates | `runs/nasa_convex_support_q_diagnostic_20260826/{status,gate_decision}.json` |
| Cell/q/weight/prediction tables | `runs/nasa_convex_support_q_diagnostic_20260826/{all_cells,all_convex_q,all_convex_weights,all_query_predictions}.csv` |
| Readable report | `runs/nasa_convex_support_q_diagnostic_20260826/DIAGNOSTIC_REPORT.md` |

## 38. Completed support-box q diagnostic (2026-08-26 21:14 CST)

The next sequential-development protocol is frozen in `NASA_SUPPORT_BOX_Q_DIAGNOSTIC_PLAN_20260826.md`. It removes the convex softmax/anchor-selection confound: each already audited, support-calibrated structure-validation q is changed by only a coordinate-wise clip to the minimum/maximum of the same cell's eight support-matched meta-fit q anchors. There is no fitted hyperparameter, new access to targets, or decoder change. Because prior outcomes on these 30 inner cells motivated the hypothesis, this can select a repair for later evaluation but is not independent confirmation.

The four frozen gates retain the exact convex diagnostic criteria: 30/30 finite integrity with zero upstream leakage and box violation, all continuity raw shifts at most 3.0, continuity functional median at most 3.0 with at least 12/15 cells at most 6.0, and prediction retention within 5% in median with at least 10/15 per-cell ratios at most 1.10. Passing would authorize only a separately frozen bounded symbolic Stage C2; failure ends this exact zero-margin box without widening it on these cells.

Compilation, CLI checks, `git diff --check`, and 56 tests passed before execution. At the pre-smoke refresh, the extended campaign remained terminal, no related process/tmux session existed, the formal output root was absent, and physical GPUs 0, 6, and 7 were empty; cards 1--5 were occupied by unrelated work. A non-counted continuity inner0/seed0 smoke on GPU 0 passed structural checks: 8/5 entities, zero upstream leakage and box violation, 55% coordinates clipped, raw/functional max-|z| 2.192/1.748. Prediction worsened from NRMSE 1.6294 to 2.0083 (ratio 1.233); this is an unfavorable single-cell observation, not the frozen 15-cell gate decision.

The formal matrix completed 30/30 with **3/4 gates passed** and does not advance. Continuity raw max-|z| has median/maximum 2.276/2.547 and functional max-|z| 2.436/5.549, so geometry and all 15 functional-tail cells pass. Prediction retention fails: median NRMSE is 1.702 versus 1.193 unconstrained, median ratio 1.208, and only 6/15 cells remain within 10%. MSE gives 1.145 versus 0.9175, ratio 1.243, and 6/15 retained. Median coordinate clip fraction is 0.60 for both losses.

Durable conclusion: two materially different post-training constraints—convex mixtures and minimum-change coordinate boxes—both eliminate q coordinate extrapolation but produce nearly the same prediction penalty and split-specific behavior. Do not tune a wider box or convex regularizer on these cells. The stronger diagnosis is that the decoder was trained to use full-curve embeddings but evaluated with prefix-only inferred q; post-hoc coordinate repair cannot change that learned dependency. The next justified candidate must align the q information pathway during training.

Source-of-truth artifacts: `NASA_SUPPORT_BOX_Q_DIAGNOSTIC_PLAN_20260826.md` and `runs/nasa_support_box_q_diagnostic_20260826/{status,gate_decision,all_cells,method_summary,DIAGNOSTIC_REPORT}.{json,csv,md}` as applicable.

## 39. Completed information-matched prefix-q training pilot (2026-08-26 21:37 CST)

The frozen sequential-development plan is `NASA_PREFIX_Q_TRAINING_PILOT_PLAN_20260826.md`. The new isolated methods `prefix_q_mse_step1` and `prefix_q_continuity_step1` use alternating blocks: q is updated first from only the earliest 30% rows per training entity, then theta is updated from the complete batch with q frozen. Continuity response-distance targets are also computed only from prefix rows. Per batch theta and q each receive one step, matching old step1 per-parameter update counts, while two separate backward passes expose the added compute honestly.

The matrix is the same three 8/5 inner splits × five seeds × two losses at q=4 and 1,000 epochs. Advancement requires 30/30 integrity, continuity prediction retention within the old support-matched comparator, raw/functional interface safety, and cross-seed q/functional stability. Because these inner cells have already informed method design, even a pass is development evidence and authorizes only bounded Stage C2, not an independent claim.

The implementation adds only the new prefix q-training fields/path and two method names. A dedicated unit test verifies q-first/theta-second phase order, equal block steps, two backward passes, and 26 processed examples for a 20-row two-entity example with six prefix rows. Full verification is 57 tests passed with the same two expected small-sample R2 warnings; compile, `git diff --check`, and a 30-cell dry-run also passed. A non-counted five-epoch GPU smoke passed with exact config (`alternating`, ratio 0.3, order feature 0, continuity 0.05), 15/15 theta/q steps, 30 backward passes, finite NRMSE, exact 8/5 q rows, and every checkpoint/q/prediction/geometry artifact present.

Formal training and frozen functional analysis completed 30/30 with zero launcher failures, 30 metadata records, and all aggregate tables. The frozen decision is **1/4 PASS; DO NOT ADVANCE**. Integrity passed. Prediction retention failed by the absolute-median clause: prefix continuity median NRMSE is 1.387 versus 1.193 old support-matched, although the paired ratio median is favorable at 0.968 and exactly 10/15 cells are within 10%. The method improves the old cell in 10/15 direct pairs but has five large regressions, so it is a promising unstable mechanism rather than a retained predictor.

Interface safety fails: continuity train-to-validation raw-q max-|z| has median/maximum 8.815/19.436 and decoder-functional max-|z| 3.389/12.840. MSE gives 6.378/15.020 and 3.841/6.854. Representation stability partly survives: continuity q-distance cross-seed median-of-split-medians is 0.980 with a 0.945 floor; cycle-1 capacity is 0.726 with a 0.690 floor. Early fade has median-of-split-medians 0.750 but one split is -0.119, failing the frozen 0.50 floor.

Durable training-dynamics conclusion: aligning the target information set alone does not align the optimization pathway. Train q uses Adam at lr 0.001 while co-evolving with theta for roughly 3,000 q steps; held-out q uses lr 0.05 for 200+50 steps against a frozen decoder. The high cross-seed q-distance stability alongside out-of-range validation q and one unstable functional mapping is consistent with calibration-scale/gauge mismatch, not generic absence of entity information. Do not discard the 10/15 paired improvements, but do not promote the method or relax its gates.

Post-terminal exploratory tail attribution (not a frozen gate) found that continuity retention is distributed across splits at 3/5, 3/5, and 4/5. Its NRMSE ratio has no observed monotonic association with raw shift (Spearman -0.329, p=0.232) or functional shift (0.104, p=0.713). Continuity and MSE retention ratios are instead negatively associated (-0.489, p=0.064): under the >10% regression definition only 1/15 cells fail both losses, 11/15 fail exactly one, and 3/15 fail neither. Functional shifts themselves remain positively associated across losses (0.621, p=0.0134). This separates a shared geometry symptom from largely loss-specific prediction basins and warns that a calibration prior may fix scale without fully fixing decoder-training multimodality.

Source-of-truth artifacts:

| Purpose | Path |
|---|---|
| Frozen plan | `NASA_PREFIX_Q_TRAINING_PILOT_PLAN_20260826.md` |
| Raw cells and ledger | `runs/nasa_prefix_q_training_pilot_20260826/**/result.json`, `launcher_status.jsonl` |
| Functional analysis | `runs/nasa_prefix_q_training_pilot_20260826/functional_coordinate_analysis/` |
| Frozen cells/gates/report | `runs/nasa_prefix_q_training_pilot_20260826/{pilot_cells.csv,gate_decision.json,PREFIX_Q_TRAINING_REPORT.md}` |

## 40. Completed meta-selected soft q-prior diagnostic (2026-08-27 11:16 CST)

The next protocol is frozen in `NASA_META_SELECTED_Q_PRIOR_PLAN_20260826.md` using the academic-research-suite experiment-agent boundary. For each prefix-q checkpoint it scores the fixed grid `{0, 0.001, 0.01, 0.1, 1}` by leave-one-entity-out calibration on all eight meta-fit entities. Selection uses only the calibration-internal holdout inside each entity's earliest 30% support; later meta-fit targets and all structure-validation query targets are excluded. The eight selected support-calibrated meta-fit q values then define the prior population for the five structure-validation calibrations. A +123.456 query perturbation audits that selected q is unchanged.

Advancement retains the prediction/interface/stability gates from the prefix-q pilot. This stage tests whether a support-only selected soft standardized prior can remove the large calibration tails without the prediction loss of convex/box projection. It remains sequential inner-split development, not independent confirmation.

Static compilation, CLI checks, `git diff --check`, and 57 tests passed. At 21:44 CST no GPU was empty: physical 0/7 had new ~72.5 GiB VLLM workers, 6 had a ~24.9 GiB sglang worker, and 1--5 were also occupied by unrelated jobs. The formal root remained absent and no formal process was launched. A non-counted CPU smoke first exposed a summary-only DataFrame selection bug; the minimal fix changed functional-shift calculation to use the already merged selected-q table. The rerun passed: 5/5 weights scored, 8/5 entities, zero query leakage, selected weight 0, raw/functional max-|z| 3.985/1.078, and selected validation NRMSE 1.3661 versus 1.3703 prefix-q no-prior. These values are structural/early observations only. Formal execution must wait for a fresh empty-card check and must not claim a utilization-zero but memory-occupied card.

At 21:50 CST a second host-level snapshot still found no empty card: GPUs 0/1/7 each used about 72.4--72.6 GiB, GPUs 2/3 about 70.9 GiB, GPUs 4/5 about 55.8 GiB, and GPU 6 about 24.9 GiB. The utilization-zero readings on 0/1/6/7 do not make those cards available. No formal prior cell or waiting controller was started. The beginner-readable main report now contains a four-row causal comparison of support matching, convex bounding, coordinate-box bounding, and prefix-q training; compact terminal summaries for all four completed stages are explicitly included by `.gitignore` while raw checkpoints, predictions, and logs remain local.

The verified transition was committed as `e8d4f5c` (`diagnose and align NASA latent q interface`) and pushed to `origin/research/latent-q-stagec-20260826` at 21:52 CST. The pre-commit suite was 57 passed with only the two expected small-sample R-squared warnings.

Before any formal cell, the plan recorded an execution-only resource amendment from the then-occupied GPUs 0/7 to freshly empty physical GPUs 4/5; no scientific setting changed. Continuity and MSE processes then completed 15/15 cells each with no retry or deadline. Terminal integrity was exact: 30 cells, 390 selected q rows, 150 prior-score rows, zero query-target leakage, finite outputs, and selected weights only from the frozen grid.

The frozen decision is **1/4 PASS; DO NOT ADVANCE**. Continuity selected-prior NRMSE is 1.3556 versus 1.3872 for prefix weight 0 and 1.1932 for old support matching; its paired selected/old ratio median is 0.9786 and 10/15 cells remain within 10% of old. Interface safety still fails with raw/functional max-|z| medians 8.188/3.273, despite 14/15 functional cells at most 6. Representation stability fails: q-distance median-of-split-medians/minimum split is 0.758/0.461, capacity 0.655/0.655, and early fade 0.691/0.119. Continuity selects weight 0 in 12/15 cells and 0.001 in 3/15. MSE has NRMSE 1.0798, raw/functional medians 5.051/3.378, and only 8/15 prediction-retained cells.

A post-terminal, explicitly post-hoc failure diagnostic did not alter the gate or use structure-validation query to choose a new weight. Support selection loss is directionally associated with later meta-fit query NRMSE (median within-cell rank Spearman 0.900 for continuity), but it matches the meta-query oracle in only 6/15 cells and favors weak/no prior. More decisively, no fixed raw-q weight in `{0, 0.001, 0.01, 0.1, 1}` passes the existing continuity representation gate. Weight 0.001 gives q/capacity/fade median-of-split-medians 0.777/0.702/0.714 but an early-fade minimum split of 0.274; weight 0.01 raises the q minimum split to 0.792 but capacity/fade medians fall to 0.667/0.560. Therefore do not spend structure-validation queries rerunning fixed raw-q weights.

Durable conclusion: this is a coordinate-definition failure, not evidence that q lacks entity information. A raw-q Gaussian prior is expressed in each seed's arbitrary embedding coordinates and preserves the exact affine gauge between q and the decoder first layer. The next prior must operate in decoder-response/functional space. Source-of-truth artifacts are `NASA_META_SELECTED_Q_PRIOR_PLAN_20260826.md`, `runs/nasa_meta_selected_q_prior_20260826/{status,gate_decision,META_SELECTED_Q_PRIOR_REPORT,RAW_Q_PRIOR_FAILURE_DIAGNOSTIC}.{json,md}` as applicable, plus the compact CSV summaries in that root.

## 41. Active functional-response prior meta-only screen (2026-08-27 11:45 CST)

The next protocol was frozen before formal cells in `NASA_FUNCTIONAL_RESPONSE_PRIOR_META_PLAN_20260827.md`. It uses only the 15 continuity prefix-q checkpoints and only their eight meta-fit batteries. For each leave-one-out battery it regularizes the decoder's normalized responses at the four already frozen conditions `(cycle, ambient, load, cutoff)` = `(1,24,2,2.5)`, `(10,24,2,2.5)`, `(20,24,2,2.5)`, and `(28,24,2,2.5)`. Per-probe response standard deviations are floored at 0.05. The raw-q prior is zero and the functional weight grid remains `{0, 0.001, 0.01, 0.1, 1}`.

Phase A never reads structure-validation data. Later 70% meta-fit cycles score prediction development only; a +123.456 perturbation must leave candidate q unchanged. A weight is eligible only if its 15-cell median meta-query NRMSE is at most 1.05 times weight 0 and it passes the existing q/capacity/fade stability gate. The lowest eligible meta-query NRMSE is selected, ties preferring the smaller weight. No eligible weight means STOP; an eligible weight authorizes only a separately frozen Phase B.

The core calibration path now accepts `calibration_functional_prior_weight` and explicit probe features; the default is zero, so existing behavior is unchanged. The new runner and analyzer are `scripts/run_nasa_functional_response_prior_meta_20260827.py` and `scripts/analyze_nasa_functional_response_prior_meta_20260827.py`. A non-counted CPU smoke on inner0/seed0/weight 0.001 passed: one cell, eight leave-one-out candidate q values, zero leakage, finite support loss and meta-query NRMSE, plus capacity/fade and all four probe responses. Full verification is 58 tests passed with the two expected tiny-sample R-squared warnings.

No formal Phase-A root exists yet. At 11:40 CST the sandbox GPU query could not access the driver; a host-level query at 11:42 CST showed every physical card occupied: memory usage for GPUs 0--7 was approximately 13.4, 22.9, 57.6, 57.6, 57.6, 57.6, 76.4, and 17.0 GiB. GPUs 2--5 belonged to one four-card sglang job. Do not launch on utilization-zero cards with resident memory. A takeover should refresh host `nvidia-smi`; only if a card has no foreign process and effectively zero memory should it run the exact command in the frozen plan, then run the analyzer after 15/15 terminal success. No automatic retry, deadline, structure-validation run, or Stage C2 is authorized.

The compact raw-q terminal evidence, failure diagnosis, functional-response Phase-A implementation, frozen plan, tests, and beginner-readable report update were committed as `0ebf0ff` (`diagnose raw q prior and freeze response prior screen`). Raw per-entity predictions, checkpoints, and logs remain ignored; manifests, exact gate inputs, aggregate q/score tables, and both readable reports are tracked.

## 42. Completed fixed-probe functional-prior screens (2026-08-27 21:48 CST)

The mean-response prior in `NASA_FUNCTIONAL_RESPONSE_PRIOR_META_PLAN_20260827.md` completed 15/15 continuity meta-fit cells with zero failures and exact zero query-target leakage. No fixed weight passed the frozen representation gate, so structure validation was not read. Weight 0 had pooled meta-query NRMSE `0.07235`; weights `0.001/0.01/0.1/1` gave `0.08917/0.1142/0.1774/0.2341`. Increasing the mean prior compressed legitimate between-entity response variation while raw-q distances expanded, consistent with a nonlinear decoder moving q farther to resist over-shrinkage. This exact mean prior is stopped.

The independently frozen rank-2 response-subspace repair in `NASA_FUNCTIONAL_SUBSPACE_PRIOR_META_PLAN_20260827.md` also completed 15/15 with zero failures and leakage. It preserved the two leading standardized four-probe response directions and penalized only the orthogonal residual. Prediction and response geometry were retained: weights `0/0.001/0.01/0.1/1` produced meta-query NRMSE `0.07235/0.07254/0.07193/0.07941/0.07052`; weight 1 improved the development median by 2.53% and had response-geometry median/minimum-split `0.789/0.631`. Nevertheless no weight passed the frozen named-coordinate gate because the fixed-condition early-fade minimum split remained low (`0.143` at weight 1). The formal decision is STOP and no structure-validation target was read.

Durable interpretation: a full response-mean prior is too restrictive, while a low-rank functional-subspace prior is technically viable and retains both prediction and gauge-invariant response geometry. The remaining failure must be diagnosed at the physical-probe level before rejecting the representation. Source artifacts are the two frozen plans and `runs/nasa_functional_{response,subspace}_prior_meta_20260827/`, especially each root's `manifest.json`, `status.json`, `weight_eligibility.csv`, `selected_weight.json`, and readable report.

## 43. Protocol mismatch diagnosis and active corrected screen (2026-08-27 21:59 CST)

A post-terminal no-retraining audit found a concrete measurement confound. The frozen reference `(24°C, 2A, 2.5V)` occurs in 0/716 rows and 0/8 labels in inner1, and only 168/711 rows in each of inner0 and inner2. Re-evaluating the already saved rank-2 q candidates at each battery's first observed protocol uses input features only and changes no q. At weight 1, cross-seed capacity stability recovers from fixed-probe `0.631/0.619` to protocol-matched `1.000/0.976`; early fade recovers from `0.655/0.143` to `0.893/0.833` under the frozen median-of-split-medians/minimum-split-median estimand. Protocol-matched four-point response-geometry worst individual seed-pair correlations are at least `0.875` in all splits. Across the 15 meta-fit cells, the decoder-derived capacity coordinate has median Spearman `0.976` with an empirical cycle-1 descriptor; early fade has `0.690` at weight 1 and `0.786` at weight 0. These descriptor correlations are explicitly post-hoc and use meta-fit targets, but they provide direct development evidence that the stable functional coordinate is scientifically aligned. This supports a probe-setting failure, not loss of functional q semantics.

The reproducible post-hoc audit is `scripts/analyze_nasa_protocol_matched_functional_stability_20260827.py`; its outputs are under `runs/nasa_functional_subspace_prior_meta_20260827/PROTOCOL_MATCHED_FUNCTIONAL_DIAGNOSTIC.md` and the adjacent protocol-matched CSV files. It cannot retroactively alter the completed frozen decision.

The corrected sequential-development screen was frozen before execution in `NASA_PROTOCOL_MATCHED_FUNCTIONAL_SUBSPACE_PLAN_20260827.md`. For each held-out meta-fit battery, cycles 1/10/20/28 are evaluated at the ambient/load/cutoff values of its first observed discharge; no target chooses the protocol. The rank-2 residual prior, fixed weight grid, support/query boundary, perturbation audit, and preceding response-based thresholds are unchanged. A single non-counted inner0/seed0/weight-0.001 smoke passed with eight finite candidates and zero leakage. Full verification before launch was 58 tests passed with the two expected small-sample R2 warnings.

At 21:59 CST physical GPUs 2--5 were empty (4 MiB, no compute PID); GPU 2 was selected after a host-level refresh. The frozen plan hash is `1880b86e10ff640c38c3b4248765e3303ed8433e5f1ba4c695f287027baf99ee`, runner hash `fa62bca54ac238a18adc466424444968f8f7548e061fa8591fcf9dc5263d5313`, protocol `first-observed`, and subspace rank 2.

An orchestration failure occurred before a trustworthy terminal run: the initial tmux task remained alive on the host after its session became invisible inside the sandbox, and a diagnostic `--resume` process created a second writer. Duplicate status lines exposed the collision after four unique cells. Both owned processes were stopped; no foreign job was touched. The full conflicting directory is preserved, not deleted, at `runs/nasa_protocol_matched_functional_subspace_20260827_orchestration_collision_20260827_2206/` and must never be used as scientific evidence. At approximately 22:07 CST a clean single writer was started from an absent exact formal root `runs/nasa_protocol_matched_functional_subspace_20260827` on GPU 2. Its live unified session identifier is environment-local and should not be relied on by a takeover; refresh the process, root, and GPU directly. If and only if the clean analyzer selects an eligible weight, freeze a separate outer-validation protocol before reading structure-validation outcomes.

## 44. Completed clean protocol-matched meta screen (2026-08-28 09:46 CST)

The clean replacement run completed 15/15 unique cells with exactly 15 append-only status rows, zero failures, 600 candidate q rows, finite outputs, and maximum query-target leakage difference zero. It used only the eight meta-fit batteries in each split; `selection_uses_structure_validation` is false. The frozen functional-response analyzer passed integrity and authorized Phase B.

Four weights were eligible. Weight 0 had pooled meta-query NRMSE `0.072349`; weight `0.001` was numerically the same at displayed precision; weight `0.01` achieved the frozen minimum `0.07136`; weight `0.1` failed prediction retention at `0.07706`; weight 1 was eligible at `0.07295`. Therefore the immutable Phase-B choice is `0.01`, a 1.37% development improvement versus weight 0. At weight `0.01`, protocol-matched response-distance stability is `0.994/0.936`, capacity is `1.000/1.000`, and early fade is `0.821/0.786`, where pairs denote median-of-split-medians/minimum split median. Raw-q stability remains low but is not gauge invariant and was not the frozen gate.

Source of truth: `runs/nasa_protocol_matched_functional_subspace_20260827/{manifest,status,selected_weight,weight_eligibility}.json` or CSV as named, plus `FUNCTIONAL_RESPONSE_PRIOR_META_REPORT.md`. Never substitute the preserved orchestration-collision directory.

## 45. Active frozen protocol-matched Phase B (2026-08-28 09:51 CST)

Phase B was frozen before formal structure-validation output in `NASA_PROTOCOL_MATCHED_FUNCTIONAL_PRIOR_PHASEB_PLAN_20260828.md`. It compares only weights 0 and `0.01` on five structure-validation batteries per split for all 15 split/seed cells. Each q sees only the earliest 30% targets; protocol features come from the battery's first observed input row; later targets are evaluation-only and receive the +123.456 leakage audit. Rank remains 2 and no raw-q prior is used. The four frozen gates are integrity, prediction retention, functional stability/response-geometry retention, and empirical capacity/fade alignment. These batteries are held out for this protocol but were exposed by older project experiments, so this is not a globally untouched final test.

Compilation, `git diff --check`, and 58 tests passed before execution. A non-counted inner0/seed0 smoke produced 5 labels × 2 weights, ten finite q rows, zero leakage, and a selected/baseline NRMSE ratio `0.99591`. At the formal launch, GPUs 2--5 were empty at 4 MiB with no compute process; GPU 2 was chosen. The exact formal root was absent. The single foreground writer started at 09:50 CST under plan hash `4fe0f492cc4512c92f2b9bc4afe9395bd46936ab602ddbaae297708363912564` and runner hash `8f0225e07413c23427d7c2ffb5bc3e3ad2a11db98e789feb14e17e506611d1a5`. At this timestamp 1/15 cells is complete with zero leakage. Refresh `runs/nasa_protocol_matched_functional_prior_phaseb_20260828/status.jsonl`, process ownership, and GPU state before reporting or intervening.

## 46. Completed protocol-matched Phase B and failure attribution (2026-08-28 09:58 CST)

Phase B completed 15/15 with exactly 15 status rows, no failures, 150 validation q rows, two fixed weights in every cell, and zero query-target leakage. The immutable selected weight remained `0.01`. Frozen prediction retention passed: selected/baseline median-NRMSE ratio is `1.00085`, all 15/15 cells remain within 10%, and the paired Wilcoxon p-value is `0.7615`. Per-split baseline versus selected median NRMSE is inner0 `1.3872/1.3884`, inner1 `1.4754/1.4751`, and inner2 `1.1543/1.1530`. Do not claim a prediction improvement; claim retention.

Scientific alignment passes after correcting only binary floating comparison at the exact Spearman boundary: capacity/early-fade median alignment is mathematically `1.0/0.5`; the raw float for 0.5 is `0.49999999999999994`, so the analyzer uses a `1e-12` comparison tolerance. Functional stability fails only because early-fade split medians are inner0 `0.85`, inner1 `0.90`, inner2 `0.20`, below the frozen 0.50 floor. Capacity stability is `1.0/1.0`. Four-point response geometry is retained in every split and improves on inner2 (`0.921` selected versus `0.873` baseline). Final frozen decision is **3/4 PASS, STOP before Stage C2**.

Post-terminal diagnosis is reproducible in `scripts/analyze_nasa_phaseb_fade_failure_20260828.py` and `runs/nasa_protocol_matched_functional_prior_phaseb_20260828/FADE_FAILURE_DIAGNOSTIC.md`. Weight 0 has the same inner2 fade stability `0.20`, so weight `0.01` did not cause the failure. Early fade is a small difference between two large decoder responses: across splits, median within-battery seed SD is 36--58% of between-battery fade spread, versus only 3--4% for capacity. B0036, B0039, and B0033 have first-cycle recovery/activation transients; B0039/B0040 also switch protocol inside 28 cycles. Thus cycle-1-to-10 slope is not uniform across these records. Rank 2 deliberately retains the fade direction, so it cannot canonicalize this noise inside its preserved subspace.

## 47. Active protocol-matched rank-1 meta screen (2026-08-28 10:03 CST)

The next minimal hypothesis was frozen before rank-1 formal output in `NASA_PROTOCOL_MATCHED_RANK1_PRIOR_META_PLAN_20260828.md`: preserve only the dominant capacity direction and softly regularize orthogonal response-shape directions. Every other protocol, weight, support/query boundary, perturbation audit, and functional/response threshold is unchanged from the clean rank-2 meta screen. This is sequential development informed by Phase B, not independent evidence.

Compilation and the full 58-test suite passed. A non-counted inner0/seed0/weight-0.01 smoke returned eight finite q values, zero leakage, and meta-query NRMSE `0.05600`. At a fresh host refresh GPUs 2--5 remained empty and the exact formal root was absent. The single foreground writer started on GPU 2 under plan hash `afb1fc9c9efbf5ca91dc13f2156b06f7de799531466cf7b633aa9807554c3fa4` and runner hash `fa62bca54ac238a18adc466424444968f8f7548e061fa8591fcf9dc5263d5313`. Refresh `runs/nasa_protocol_matched_rank1_prior_meta_20260828/`, the process, and GPU ownership before reporting.

## 48. Completed rank-1 meta and development replication (2026-08-28 10:52 CST)

The rank-1 meta screen completed 15/15 with exact integrity and zero leakage. It selected weight `0.01`: meta-query NRMSE `0.07041` versus `0.07235` at weight 0, a 2.68% development improvement. Protocol-matched response/capacity/early-fade stability is `0.991/0.940`, `1.000/1.000`, and `0.857/0.821`; the latter improves rank 2 (`0.821/0.786`). This validates the meta-level mechanism that preserving only the dominant capacity direction can stabilize residual response shape.

The separately frozen same-cohort development replication in `NASA_PROTOCOL_MATCHED_RANK1_PHASEB_DEVELOPMENT_PLAN_20260828.md` completed 15/15 with zero leakage but stopped at **2/4 PASS**. Prediction retention narrowly fails: selected/baseline median-NRMSE ratio `1.05162` exceeds 1.05, although 14/15 cells are within 10%. Per-split baseline/selected medians are inner0 `1.387/1.459`, inner1 `1.475/1.474`, and inner2 `1.154/1.153`. Functional stability still fails because early-fade minimum split remains `0.20`; capacity remains `1.000/0.900`. Scientific alignment passes at capacity/fade `1.0/0.5`.

Durable conclusion: do not tune subspace rank further on these cohorts. Rank 1 improves exposed meta-fit endpoints but does not transfer the fade repair and slightly harms outer prediction, concentrated in inner0. Rank 2 is preferable for prediction retention and full response geometry, while neither rank makes cycle-1-to-10 fade a uniform robust scalar. The confirmed real-data positive endpoint is protocol-matched capacity: near-perfect cross-seed stability and empirical alignment on held-out batteries. A next symbolic stage should be independently frozen around capacity plus the bounded four-response interface, with early fade removed from the primary vocabulary rather than silently excluding failed batteries. Any final confirmation needs new batteries or another real dataset.

## 49. Active MATR cross-batch acquisition and confirmation design (2026-08-28 13:17 CST)

The user authorized finding additional datasets, using available GPUs, and using subagents where helpful. A local audit, an official-source dataset search, and an adversarial experiment-design review converged on the MATR/Severson LFP fast-charge cohorts as the fastest defensible new real-battery evidence. The existing repository contains only the historically exposed 2017-05-12 file in prepared experiments. The new plan is frozen in `MATR_CROSS_BATCH_CONFIRMATION_PLAN_20260828.md`: Batch1 is development, Batch2 may freeze choices, Batch3 is a sealed 40-cell secondary confirmation cohort, and the 2019 Batch4 file is identity-audit only until its provenance is resolved. PulseBat is next in the fixed candidate order, followed by NIST adsorption isotherms and the NASA/UCF randomized/recommissioned battery dataset. C-MAPSS remains excluded from all materials.

Official sources materially changed the preprocessing requirement. The authors' `LoadData.m` says five Batch2 records continue the first five Batch1 cells; a filename-only concatenation would therefore create false entities. It also declares Batch1 unfinished-cell exclusions and ordered Batch3 collection/noise/unfinished rules. The dedicated preparer `scripts/prepare_matr_cross_batch_20260828.py` implements that source-documented identity ledger, exact curve hashes, cycle/protocol/finite audits, and a fixed first-100-support eligibility check. It has compiled and passed `git diff --check`, but has not run because downloads are incomplete.

At 13:15 CST, GPUs 2--5 were empty at 4 MiB each; GPUs 0, 1, 6, and 7 remained occupied and must not be touched. Formal GPU work has not started because the new cohorts must pass acquisition and eligibility gates first. Three owned tmux downloads were launched from the official MATR endpoints: `matr_b2_download_20260828` (2,007,331,155 bytes expected), `matr_b3_download_20260828` (3,236,690,412 bytes), and `matr_b4_download_20260828` (2,601,295,745 bytes). Their targets are ignored raw files under `data/application/battery_matr/raw/`. Refresh tmux, exact file sizes, and curl ownership before intervening; `--continue-at -` permits a recoverable restart but there is no automatic retry. After all three exact sizes arrive, run the preparer once into an absent output root, reconcile the expected 41/43/40 formal cell structure with the source ledger, then implement and test the exact 100-row prefix boundary before any model smoke.

The local branch is still one commit ahead of its remote at `e1027d2`. A prior push attempt was blocked by the approval reviewer because the exact new commit scope had not been explicitly reapproved. Do not retry or bypass that review without the user's explicit approval to push the scoped local commit.

At 13:24 CST the exact-count prefix implementation was complete. `LatentQConfig` now has optional `calibration_count` and `q_training_count`; the real-data runner exposes `--support-count`, carries it into the immutable job/manifest payload, uses it for every baseline support/query split, and applies it to both training-q and unseen-q calibration. An explicit count must leave at least one query row and overrides the ratio. Two regression tests cover exact splitting and q-phase example accounting; the full suite is 60 passed with only the two known small-sample R2 warnings.

A non-counted old-Batch1 GPU smoke on physical GPU 2 completed successfully at `runs/_smoke_matr_support100_20260828/`. It used two epochs and two calibration steps only. All 12 test cells contributed exactly 100 support rows (1,200 total), leaving 9,024 query rows; the saved job records `support_count: 100` and `q_training_split_mode: prefix`. This validates plumbing only and must never enter a result table or hyperparameter decision. GPU 2 was released after the smoke. Formal training remains blocked on the source-identity audit, not on code execution.

## 50. Active capacity-bounded symbolic Stage C2 and acquisition refresh (2026-08-28 13:50 CST)

A read-only audit of the rank-2 meta and Phase-B artifacts confirmed a material flaw in the old Stage C information path: old symbolic-fit q came from full-curve `train_label_q.csv`, while structure-validation q came from prefix-only support calibration. The new sequential-development plan `NASA_CAPACITY_BOUNDED_SYMBOLIC_STAGEC2_PLAN_20260828.md` fixes both sides to prefix-calibrated q at the globally selected rank-2 weight `0.01`. The runner now machine-checks `selected_weight.json` integrity, exact weight equality, and `selection_uses_structure_validation == false`. Its sole q-derived feature is the protocol-matched `capacity_cycle1`, clipped to the eight meta-fit batteries' median ±3 IQR; raw q, early fade, division, logarithms, exponentials, and square roots are excluded. The 45-cell matrix is 3 inner splits × 5 seeds × 3 equal-budget interfaces: condition-only, condition plus robust prefix summaries, and condition plus bounded capacity q.

Static compilation and a dry-run passed. The dry-run audited all 15 split/seed blocks, exact 8/5 entity isolation, prefix ordering, and maximum validation-query-target symbolic-input difference `0.0`. Plan SHA-256 is `0274b4f152bf9b201c45c6116d29b2b9dca115ad6f1aecb469a08e0e798f5bb2`; the current runner SHA-256 is `600df985fc702a6ea542bb16e4620fdb29a4f39cc8acb8a095defaa3af04cecb`. A non-counted 2-iteration PySR 1.5.10 smoke on inner0/seed0/capacity-q completed with finite predictions and a physical-unit expression; its low-budget formula used only cycle and is plumbing evidence only.

The first formal launch exposed a protocol mismatch before any result was promoted: the runner still had a 7,200-second per-cell PySR timeout despite the frozen no-time-limit contract and the user's run-until-complete instruction. That owned session was stopped, no foreign process was touched, and the partial root was preserved as `runs/nasa_capacity_bounded_symbolic_stagec2_20260828_aborted_timeout_contract_1349/`. The default timeout is now `None`. At 13:50 CST a clean single writer was launched in tmux `lvs_nasa_capacity_stagec2_20260828` from an absent exact root `runs/nasa_capacity_bounded_symbolic_stagec2_20260828`; its manifest records `job_timeout_seconds: null`, 80 iterations, maximum complexity 14, 1,200 fit rows, and two parallel CPU jobs with two Julia processes each. It consumes no GPU. After 45/45 terminal success, run `scripts/analyze_nasa_capacity_bounded_symbolic_stagec2_20260828.py`; do not authorize the structured-decoder refit unless all five frozen gates pass.

The host GPU refresh at 13:45 CST showed cards 2--5 empty at 4 MiB, while 0, 1, 6, and 7 were occupied. MATR Batch2 and Batch3 official downloads remain active in owned tmux sessions. Batch4's original session exited after leaving a 295,412,548-byte partial file; no residual Batch4 curl existed. At 13:48 CST one explicit recoverable continuation was started in `matr_b4_download_20260828` using the already audited official URL and `--continue-at -`. Exact expected sizes remain 2,007,331,155 / 3,236,690,412 / 2,601,295,745 bytes for Batch2/3/4. Do not run the cross-batch preparer or claim a GPU until exact sizes and current ownership are refreshed.

## 51. Completed capacity-bounded Stage C2 and authorized next diagnosis (2026-08-28 14:04 CST)

The clean no-time-limit Stage C2 completed 45/45 cells with zero failures. Integrity passed exactly: 45 result/prediction/Pareto/scaler artifacts, exact 8/5 entity isolation, selected rank-2 weight `0.01`, all finite metrics and predictions, physical-unit formulas, and maximum query-target symbolic-input difference `0.0`. The full 60-test suite also passed with only the two known small-sample R2 warnings.

The frozen scientific decision is **2/5 PASS** and `authorize_capacity_anchored_structure=false`. Readability and integrity pass; held-out value, information-matched value, and motif recurrence fail. Pooled median structure-validation NRMSE is `0.865778` for capacity-q, `0.865780` for condition-only, and `0.865826` for robust support summaries, but capacity-q has a worse mean `0.919893` because of an inner1 tail. It wins only 7/15 against condition-only and 7/15 with one tie against support summaries. Split medians for capacity-q / condition / support are inner0 `0.751953/0.751969/0.751969`, inner1 `1.316390/0.984971/0.984979`, and inner2 `0.865827/0.865780/0.865826`. Only 5/15 selected formulas use capacity and 4/15 have a capacity-conditioned cycle slope, below 12/15 and 10/15.

All five capacity-using formulas occur in inner1. Two seeds are genuine favorable development cells (`0.85416` and `0.78507` versus roughly `0.98498`) and learn capacity-conditioned cycle/load motifs, while three seeds degrade to `1.316--1.399`. These favorable seeds must not be selected alone. Entity diagnosis points to fixed cycle-1 probe mismatch on activation/recovery batteries: for inner1 seed0, B0039 has support tail-10 median about `0.465 Ah` but the first query target about `1.772 Ah`; its decoder cycle-1 coordinate is about `1.401 Ah`, causing a negative population coefficient. The bounded grammar removed the old Stage-C catastrophic extrapolation, so the remaining blocker is a functional-coordinate/task-boundary mismatch rather than raw-q discontinuity.

The reproducible outputs are `runs/nasa_capacity_bounded_symbolic_stagec2_20260828/{gate_decision,integrity_audit,motif_summary}.json`, `method_summary.csv`, `split_summary.csv`, `paired_comparisons.csv`, `capacity_motif_cells.csv`, all raw per-cell artifacts, and `STAGEC2_REPORT.md`. A read-only implementation audit confirmed Stage D must remain stopped. The authorized next sequential-development experiment is a newly frozen Stage C2b comparing the same q evaluated at cycle 1 versus the first-query support boundary, a pure support anchor, and a support-anchor-plus-boundary-q residual interface. It must keep all batteries and all five seeds, use the same decoder/q checkpoints, expose all outcomes, and remain development-only. MATR Batch3 remains untouched confirmation.

The MATR preparer now has an exact byte-size gate at the external-file boundary for all four official files. A live partial-file invocation failed fast before creating any output, reporting the observed and expected sizes; this prevents partial downloads from entering HDF5 parsing or GPU training.

## 52. Active deterministic support-boundary Stage C2b (2026-08-28 14:31 CST)

The next development experiment is frozen in `NASA_BOUNDARY_CAPACITY_SYMBOLIC_STAGEC2B_PLAN_20260828.md`. It keeps all 15 split/seed blocks and compares four interfaces: direct cycle-1 decoder capacity, decoder capacity at the first-query support boundary, the observed support tail-10 anchor, and support anchor plus `boundary_q - support_anchor`. Total size is 60 PySR cells. All coordinates come from the same rank-2 weight-0.01 prefix-calibrated q and frozen checkpoints. Boundary inputs use the first query row's known cycle/ambient/load/cutoff only; query targets do not define q, coordinates, bounds, scalers, or formulas. Every coordinate is clipped using its own eight-meta-fit median ±3 IQR.

The preparatory dry run produced exactly 195 coordinate rows (`15 × (8+5)`), exact 8/5 isolation, strict support/query ordering, upstream meta and validation q leakage maxima `0.0`, and local boundary-coordinate perturbation difference `0.0`. The plan hash is `856863df20c0322bf6c66d4e9da1b490a39c66014a7eca80c94ca14698569895`; the formal runner hash is `f308b0a1f8170c58ae25de2f8f9a3c94eb38e816e1f5d7803fee72865b39943a`. PySR is deterministic serial within each cell, with up to four cells scheduled concurrently, 80 iterations, maximum complexity 14, and no timeout.

Two non-counted 2-iteration smoke runs were used to repair and verify orchestration. The first passed numerically but exposed a torch-before-Julia fork warning. The runner now uses spawn workers and does not import torch in symbolic workers before PySR. The clean second smoke passed without that warning, and its formula, complexity, and metrics exactly matched the first smoke. Neither smoke is scientific evidence.

At 14:30 CST the exact formal root was absent and the 128-core host load was about 23. The single writer was launched in tmux `lvs_nasa_boundary_stagec2b_20260828`; its manifest records 60 planned cells, `non_counted_smoke: false`, `job_timeout_seconds: null`, and deterministic serial PySR. Run `scripts/analyze_nasa_boundary_capacity_symbolic_stagec2b_20260828.py` only after 60/60 terminal success. Stage D remains unauthorized unless all five C2b gates pass.

MATR Batch2 reached its exact expected size `2,007,331,155` bytes and its tmux session exited normally. Batch3 and Batch4 remain active partial downloads and must reach `3,236,690,412` and `2,601,295,745` bytes before the cross-batch preparer runs.

## 53. Completed deterministic Stage C2b; Stage D remains stopped (2026-08-28 14:35 CST)

Stage C2b completed 60/60 deterministic cells with zero failures. Integrity passed: 60 result/prediction/Pareto/scaler artifacts, 195 finite coordinate rows, exact 8/5 isolation, exact selected weight, strict prefix boundary, physical-unit formulas, upstream q leakage maxima `0.0`, and local boundary-coordinate perturbation difference `0.0`.

The frozen decision is again **2/5 PASS** and `authorize_structured_decoder_stage_d=false`. Integrity and safety/readability pass; task-boundary coordinate value, incremental q beyond support, and symbolic recurrence fail. Pooled median NRMSE is essentially tied at `0.865780` for boundary q, direct cycle-1 q, and support anchor, while support-plus-q-residual is `0.865826`. The means reveal the tails: boundary q `1.39064`, direct q `1.05637`, support anchor `0.86758`, and q residual `1.29454`. Boundary q beats direct q in only 5/15 cells with four ties; the residual interface beats support anchor in 6/15. Only 5/15 residual formulas use q residual, 4/15 have a residual-conditioned cycle slope, and all occur in one split.

The boundary probe repairs one aspect of inner1 but not the representation. Inner1 median direct q improves from `1.52334` to `0.984979` with boundary q, nearly matching support anchor `0.984971`; however boundary q has seed-specific tails up to `5.65675`. The residual interface has one favorable seed (`0.76922`) but a median `2.32438` and maximum `3.75902`. This reproduces the earlier lesson that small differences between noisy decoder responses amplify calibration noise. The prior non-deterministic Stage C2 favorable inner1 seeds do not replicate under deterministic PySR selection, so they cannot motivate Stage D.

Source of truth is `runs/nasa_boundary_capacity_symbolic_stagec2b_20260828/`, especially `integrity_audit.json`, `gate_decision.json`, `method_summary.csv`, `split_summary.csv`, `paired_comparisons.csv`, `motif_summary.json`, `cell_diagnostics.csv`, and `STAGEC2B_REPORT.md`. The next scientifically justified move is not another post-hoc scalar probe. Either use entity-held-out structure selection/sparse physically constrained formula discovery to control formula overfit, or move the complete loop to the better-matched new MATR cohorts once acquisition and identity audits finish. NASA remains valid failure-mechanism and gauge evidence; it is not a q prediction win.

A read-only parse of the now complete Batch2 file succeeded before formal cohort preparation: 24,920 rows, 48 raw cell records, 170/508/1,060 minimum/median/maximum rows per cell, finite targets, cycle range 1--1,060, strict increasing cycles in 48/48 records, and zero duplicate `(label, cycle)` rows. This is consistent with the expected 43 formal Batch2 entities after the five documented continuation records are reassigned to Batch1. It does not replace the all-batch identity/hash audit.

## 54. CCF-A/ICLR evidence gate and MATR continuation correction (2026-08-28 14:53 CST)

No installed package is named exactly `ccfa`. The closest applicable paper-review workflow, `research-paper-writing`, was read together with its experiment and reviewer-audit references and used alongside the existing academic-research protocol. The resulting reviewer-facing artifact is `ICLR_CLAIM_EVIDENCE_GATE_20260828.md`. It maps each proposed claim to mandatory evidence, preserves negative NASA results, makes MATR cross-batch confirmation and the real symbolic closed loop explicit blockers, and forbids unsupported title/abstract wording. Current readiness remains a strong experimental foundation with a critical real symbolic-loop gap, not submission complete.

A read-only MATR runner audit found that the existing real runner can directly provide no-q MLP, Random Forest, support kNN, exact-prefix q, q artifacts, continuity/geometry metrics, and optimizer accounting. MATR-specific Huber and irregular-curve FPCA still require implementation, while the rank-2 response prior needs only a small runner interface. CPU methods are to be matched by information, entity splits, support/query rows, and frozen tuning budget rather than artificial epochs. Their exact grids, the FPCA missing-grid rule, per-cell selection metric, separate Batch2-only manifest, and neural backward/runtime audit are now frozen in `MATR_CROSS_BATCH_CONFIRMATION_PLAN_20260828.md` before Batch2 model scoring.

The source audit then found and repaired two preparer defects before formal cohort creation. Raw Batch2 continuation segments restart `summary.cycle` at 1, and the authors' `LoadData.m` appends arrays and evaluates by array position. The preparer now assigns `global_cycle = source_cycle + destination_preappend_max_cycle`; actual Batch1/2 verification gives strictly increasing ranges 2--1851, 2--2159, 2--2236, 2--1433, and 2--1708 for the five continued cells. The raw continuation protocol strings are truncated (`80%)-3.6C` or `80%)-4C`) and were previously misparsed; continuation rows now inherit the original Batch1 protocol. The source ledger records both transformations. Compilation, a real-data five-cell audit, and `git diff --check` pass.

The corrected source-only partial cohort audit now gives exactly 41 formal Batch1 and 43 formal Batch2 cells. Batch1 has 532/840/2,235 minimum/median/maximum valid rows and Batch2 170/507/745; all 84 cells have at least 120 rows, finite target/protocol values, strictly increasing cycle, zero duplicate cycles, and zero exact cross-cohort capacity-curve hashes. This is sufficient for Batch1/2 eligibility but does not replace the pending Batch3/4 all-file audit.

Timestamped resources: Batch2 is exact at 2,007,331,155 bytes. At 14:52 CST Batch3 and Batch4 were still downloading in owned tmux sessions at 2,245,451,776 / 3,236,690,412 and 1,765,385,028 / 2,601,295,745 bytes. Host GPUs 2--5 were empty at 4 MiB; GPUs 0, 1, 6, and 7 were occupied. No GPU experiment was launched because Phase 0 remains blocked on exact Batch3/4 acquisition and all-batch identity audit. Batch3 target metrics remain sealed.

## 55. Zero-context protocol review and pre-outcome repairs (2026-08-28 15:17 CST)

A fresh Luna reviewer audited the MATR and ICLR claim plans before any Batch2 model score. It accepted the continuation offset logic but identified two critical defects. First, `C100` is inside the calibration support, so correlation with cycles 91--100 would be circular reconstruction evidence. The primary scientific coordinate is now post-support `C150`, aligned only after prediction to measured cycles 141--150; `C100` remains a support-reconstruction diagnostic. The reliability protocol is frozen as ten contiguous delete-10 support jackknives with ICC(3,1) and a within-/between-cell noise ratio, plus separate cross-seed stability. Second, FPCA originally started its grid at cycle 1 even though valid Batch1 curves begin at cycle 2, creating an all-NaN column. The runner now starts at the minimum valid training cycle, reports in/out-of-grid metrics and coverage, uses final-quartile late-life metrics, and uses the final-ten-support median for persistence.

The official source is pinned to repository commit `1ef13d27c66dc3d73affdaa008fbeba5687b2ea4`; `LoadData.m` SHA-256 is `7914333f0a963a0742d9fff340f1d4bc2ad912f1b04a236b3ae6c39fedd3623d`. The continuation ledger now records source/destination row counts, raw cycle ranges, offsets, destination bounds, and protocol hashes. A result-independent protocol plausibility gate rejects silently defaulted zero-valued policy parses. The corrected live Batch1/2 partial audit passes this gate for all 84 formal cells.

Sealed-confirmation isolation is now code-level rather than a shared manifest convention. The preparer writes a Batch1/2-only development manifest and metadata file with confirmation access forbidden; it does not create a Batch3 runner manifest. The dedicated latent-prior and CPU runners reject any role/path other than Batch1→Batch2 and assert exact prefix-100 source settings. All selected trained CPU/neural baselines must be refit on Batch1+2 before Batch3; per-cell inference aggregates seeds before battery-level statistics, uses 10,000 fixed-seed bootstraps, cellwise paired ratios, declared tie handling, and BH correction.

Two development implementations are ready. `scripts/run_matr_cpu_baselines_20260828.py` covers persistence, Huber, support kNN, fixed-seed RF, and train-only FPCA with all frozen candidates and predictions preserved. `scripts/run_matr_latent_prior_development_20260828.py` trains no new decoder: it loads each exact prefix-q checkpoint, reproduces weight-0 calibration, applies the fixed rank-2/weight-0.01 probe prior at cycles 25/50/75/100, computes C100/C150, and checks q/C100/C150 invariance to a +123.456 query-target perturbation. Their focused tests pass (`1 passed` each), including exact source prediction reproduction and the real cycle-2 FPCA boundary repair.

The full symbolic objective is pre-registered before outcomes in `MATR_SYMBOLIC_CLOSED_LOOP_PROTOCOL_20260828.md`. Conditional on the Batch2 gate, five entity-held-out folds compare condition, support-summary, and bounded functional-q formulas under a fixed `{+,-,*}` grammar. A first formula defines an explicit symbolic backbone plus measured residual family, q is recalibrated through that structure, and an identical second symbolic fit must improve at least two frozen interpretability endpoints. All choices use Batch1+2; Batch3 evaluates one hashed package once. This closes the earlier protocol-design gap but does not count as a completed loop until executed and confirmed.

The learned no-explicit-q fairness gap is now mandatory: an exact-prefix attentive CNP adapter is being implemented separately. Timestamped acquisition at 15:17 CST: Batch3 was 3,194,929,152 / 3,236,690,412 bytes and Batch4 2,544,091,972 / 2,601,295,745, both owned downloads still active. No Batch3 metric has been read and no formal GPU model has launched.

## 56. Terminal MATR Phase 0 and active Batch2 development campaign (2026-08-28 15:35 CST)

All four official raw files reached their exact expected byte sizes and were hashed before formal preparation. The source-of-truth formal output is `data/matr_cross_batch_20260828/`. Batch1/2/3 contain 41/43/40 eligible cells; every formal target is finite, every cycle sequence is strict, protocols are parseable, duplicate cycles and exact duplicate formal curve hashes are zero, and the five Batch2 continuations are represented only through the documented Batch1 append ledger. Batch4 has 45 length-eligible records but uses a distinct four-number multistage policy encoding such as `4.8-5.2-5.2-4.16`; the frozen three-field parser is incompatible. It therefore remains identity/provenance-only and is not a model cohort. A target-blind nearest-curve audit found no exact Batch3/4 duplicates; generic early-curve similarity is not sufficient to promote Batch4.

The Batch1→Batch2 development boundary is machine-enforced in the generic real runner as well as the dedicated CPU/prior/attentive runners. The generic runner now refuses shared or confirmation manifests, Batch3 paths, non-prefix support, or support counts other than 100, and records manifest/data/runner/plan hashes. Frozen neural settings are 1,000 epochs, q=4, hidden sizes `(256,128)`, five seeds, and 200-step four-start support calibration plus 50 refine-only steps. CPU grids, battery-level selection, and all candidate retention remain unchanged.

A code audit found that the new attentive CNP helper reset every per-cell DataFrame index and then used that local index into global feature tensors. This could make every training/test battery reuse the first battery's rows. The minimal correction preserves global row indices; a dedicated regression test checks the exact second-cell index range. The attentive runner also records architecture, manifest, runner, and plan hashes. Focused MATR verification is 8 tests passed. A separate runtime inefficiency in the CPU baseline fitted the identical RF once per test battery; it now fits once per frozen candidate, with no model/grid/result-definition change.

Two exact-prepared-data, non-counted two-epoch GPU smokes passed on the CUDA-12.8 environment `/public/home/wangyg/.venvs/dso-cu128/bin/python`: prefix-q used 41 training cells, 43 test cells, exactly 4,300 support and 17,227 query rows; attentive used the same boundary and had exact zero prediction change after a +123.456 query-target perturbation. These smokes validate plumbing only and are excluded from scientific tables.

The formal Batch2 development root is `runs/matr_cross_batch_development_20260828/`. The CPU baseline session `matr_dev_cpu_20260828` is active and writes only `cpu_baselines/`. The neural session `matr_dev_neural_20260828` has a ten-job manifest for no-q MLP and prefix-q continuity, five seeds each; its `<128 MiB` GPU guard is active. At 15:34 CST GPUs 2--5 were newly occupied by another user's MatterGen processes at roughly 6.5--7.7 GiB and 87--92% utilization, so the neural launcher had dispatched zero jobs and was waiting. Do not terminate those foreign jobs. Partial CPU or neural outputs are development observations only. Batch3 remains sealed: no target metric, normalization, model selection, formula selection, or confirmation manifest has been created.

At about 15:35 CST the foreign MatterGen processes released GPUs 2--5. The guard then dispatched exactly four owned jobs, one per card: no-q seed0, prefix-q seed0, no-q seed1, and prefix-q seed1. Their resident memory was 0.8--0.94 GiB per card and all four PIDs mapped to the approved CUDA environment. This transition was automatic and occurred only after the cards fell below the frozen threshold. Six neural jobs remain queued behind them.

## 57. Terminal Batch2 CPU baselines; neural development still active (2026-08-28 15:39 CST)

The frozen CPU development cell completed successfully in about 170 seconds. Integrity is exact: 47/47 candidates, 43/43 Batch2 batteries per candidate, 2,021 candidate-cell rows, five selected candidates, finite primary/late metrics, all candidate predictions preserved, and no Batch3 access. The selected Batch2 median per-cell NRMSE values are FPCA/ridge `1.10620`, Random Forest `1.49308`, Huber trend `1.63875`, support kNN `1.68468`, and persistence `1.68637`. Their median late-life NRMSE values are `2.04784`, `2.82691`, `3.10073`, `3.17799`, and `3.18641`, respectively. FPCA selected 4 components and ridge `1e-4`; RF selected 500 trees, leaf size 4, and all features. Batch2 lies entirely inside the Batch1 FPCA grid, so out-of-grid coverage is zero here; retain the frozen subgroup metric for later Batch3 even if it is empty again.

These are development-selection results and not independent performance claims. They materially raise the comparison bar: the strongest non-neural competitor is the training-cohort functional basis with test-prefix coefficient inference, not persistence/kNN. At this timestamp the first four neural jobs still had no terminal `result.json`; do not compare partial logs or decide a latent candidate until all ten neural cells, all five attentive cells, and both fixed prior interfaces are terminal.

## 58. Prefix-q global-regularizer scaling failure and frozen repair (2026-08-28 15:51 CST)

The original unnormalized 1,000-epoch prefix-continuity jobs for seeds 0, 1, and 2 all failed with `FloatingPointError: Non-finite total training loss` and produced no `result.json` or Batch2 prediction score. The launcher did not retry. The concrete implementation defect is that continuity is a complete 41-cell embedding-level objective but was added at every one of 148 raw mini-batches per epoch; prior small-cohort NASA runs had only about three batches per epoch. Thus the named weight `0.05` did not define a dataset-size-invariant epoch-level regularizer dose.

Before any successful latent Batch2 score, the plan was amended transparently. `LatentQConfig.normalize_global_regularizers_per_epoch` multiplies only full-embedding feature-orthogonality, curve-continuity, q-L2, and q-whitening terms by `current_batch_rows / training_rows`; the actual batch fractions, including the final partial batch, sum exactly to one per epoch. Prediction, Jacobian, and local smoothness terms are not scaled. No-q/attentive algorithms and all architecture, seed, epoch, batch, continuity-weight, support, and calibration settings remain unchanged. New failure messages include epoch/batch/phase/component values. Optimization counters now include outer-batch count, q-phase support row min/median/max, global-scale min/max/sum, final q magnitude, pairwise-distance dispersion, and 100-backward gradient diagnostics for corrected runs.

A Luna zero-context reviewer independently accepted epoch-normalizing the global regularizer as the minimal theory-correct repair and rejected manual retuning of the `0.05` weight from Batch2 outcomes. It emphasized that this corrects regularizer semantics but not total compute equality: prefix alternating still has separate q and theta backward passes. The paper must report actual steps/backwards/runtime and describe equal epochs/theta updates rather than equal total compute.

The reproducible Batch1-only diagnostic is `scripts/diagnose_matr_global_regularizer_scaling_20260828.py`; it never loads Batch2 targets. Its first output root has an audit-field bug and is preserved at `runs/_diagnostic_matr_global_regularizer_scaling_20260828/`: unnormalized `scale_sum_per_epoch` was reported as 1 instead of 148, although training numerics were correct. The corrected v2 root is `runs/_diagnostic_matr_global_regularizer_scaling_v2_20260828/`. Both five-epoch variants were finite, with exact 148 outer batches, q-support rows min/median/max 4/28/45, and 740 theta plus 740 q steps. Unnormalized/normalized scale sums are correctly 148/1; their initial recorded q-gradient norms were approximately 0.165/0.0296. This is training-dynamics plumbing evidence, not a prediction result.

The original session `matr_dev_neural_20260828` remains active to preserve all five no-q successes and all unnormalized prefix failures. A separate session `matr_dev_neural_scaled_20260828` is waiting for the original tmux session to exit; it has not created its result root. Only then will it launch five corrected prefix seeds under the same `<128 MiB` guard at `runs/matr_cross_batch_development_20260828/neural_epoch_normalized/`. Do not run both launchers concurrently or merge failed and corrected prefix cells under one method root.

At 15:56 CST no-q seeds 0 and 1 had completed successfully after about 1,238/1,152 seconds with Batch2 macro per-cell NRMSE `1.40421/1.51921`. These are two of five optimization seeds and are partial development observations only; both are weaker than the selected FPCA median `1.10620` but must not be summarized as the neural baseline until all seeds finish. The old launcher immediately continued its pending queue. A third serial watcher `matr_dev_attentive_20260828` is also active but only waits for the scaled-prefix tmux session to end; then its dedicated guarded launcher will run attentive seeds 0--4 at `attentive_cnp/`. Its dry run and corrected six-test runner suite passed, and the formal attentive root remains absent while waiting.

By 16:02 CST all five original unnormalized prefix seeds had failed with the same non-finite training error and zero result files. No-q seeds 0--2 had succeeded with macro per-cell NRMSE `1.40421`, `1.51921`, and `1.44889`; seeds 3--4 remained active. The uniform five-seed failure strengthens the dataset-size regularizer diagnosis but is not a negative prediction comparison because no old-prefix Batch2 prediction exists.

The fixed rank-2 prior runner now explicitly requires `normalize_global_regularizers_per_epoch=true` in every source job and records source-result/checkpoint, manifest, plan, and runner hashes. Its focused end-to-end test passes. The dedicated launcher validates that exactly one successful corrected q result exists for every seed before creating its output root. Session `matr_dev_latent_prior_20260828` waits for attentive completion and then launches the five fixed weight-0/rank2-weight-0.01 interface comparisons with the same GPU guard; a missing corrected seed causes a visible fast failure rather than fallback to old q.

## 59. Empty-card acceleration and symbolic leakage audit (2026-08-28 16:11 CST)

At 16:09 CST GPU2 still ran only owned no-q seed4, while GPUs3--5 were empty at 4 MiB. The scaled watcher, attentive watcher, and prior watcher were verified to be waiting-only with all downstream result roots absent. They were stopped in dependency-safe order and recreated so the corrected prefix launcher could immediately use only GPUs3--5 rather than wait for no-q seed4. Seeds 0--2 started one per card at approximately 0.94 GiB each; the old no-q seed4 continued untouched on GPU2. The attentive watcher again waits for the scaled session; the prior watcher waits for attentive. This is orchestration-only and changes no scientific setting or output root.

A pre-implementation symbolic audit found a potential leakage in the current written protocol: five symbolic folds over Batch1+2 would treat some Batch1 batteries as formula-held-out even though the source decoder theta saw their full curves during Batch1 training. No symbolic result exists yet. Before executing the closed loop, decide and freeze either (a) formula folds only over the 43 genuinely decoder-held-out Batch2 batteries and a fully frozen Batch1-trained package for Batch3, or (b) decoder cross-fitting inside every symbolic fold. A Luna read-only audit is comparing these options. Do not run the current B1+2 symbolic folds until this boundary is resolved.

## 60. Empty-prefix minibatch repair, attentive completion, and decoder-cross-fit freeze (2026-08-28 16:35 CST)

The five epoch-normalized prefix-continuity jobs also failed before producing any Batch2 prediction. Their q-phase prediction term became `NaN` at epochs 240--445 while the continuity component stayed finite; every failure was at the final shuffled raw mini-batch. The exact defect was an empty q minibatch: raw-row batching can yield a batch containing none of the 4,100 prefix-support rows, and the code computed a mean prediction loss over the empty tensor. This is separate from the already corrected global-regularizer dose.

The minimal repair skips only a q phase whose selected support row count is zero and records `q_phase_empty_batches_skipped`; all nonempty q updates, theta updates, learning rates, losses, seeds, data, and calibration remain unchanged. A deterministic batch-size-one regression test verifies 40 theta steps, four q steps, 36 skipped empty q phases, and finite training over two epochs. Three focused pipeline tests pass under the CUDA environment through `unittest`. The plan now transparently records both failed roots and this implementation repair. The training-dynamics note also contains the earlier epoch-dose analysis; synchronize the empty-batch mechanism there before paper use.

At 16:31 CST GPUs 2--5 were empty at launch and a third isolated five-seed formal root started in tmux `matr_dev_neural_nonempty_20260828`: `runs/matr_cross_batch_development_20260828/neural_epoch_normalized_nonempty/`. Seeds 0--3 were dispatched once; seed 4 remains queued. By 16:34 CST unrelated VLLM processes had joined GPUs 2 and 3 after dispatch (about 53 GiB and 7 GiB respectively), while the four owned q jobs remained below 1 GiB each. Do not touch the foreign jobs. The guarded launcher will not dispatch seed 4 onto a card above 128 MiB. No corrected result or status row existed at this timestamp.

Attentive CNP is terminal: five/five seeds succeeded with five zero-return-code ledger rows and query-target perturbation difference zero. The fixed rank-2 prior runner now additionally emits ten contiguous delete-10 C150 jackknives for both latent interfaces; its focused synthetic end-to-end test passed. Waiting tmux `matr_dev_latent_prior_nonempty_20260828` will start only after the corrected q session exits and requires exactly one successful source per seed. Its formal root is `latent_prior_nonempty`, not the earlier absent/failed-source `latent_prior` root.

The symbolic leakage audit selected the stricter ICLR-facing option: five decoder-cross-fit folds over all 84 Batch1+2 batteries. `MATR_SYMBOLIC_CLOSED_LOOP_PROTOCOL_20260828.md` now requires every formula-validation entity to be absent from decoder training, support-recalibrated q for both formula-train and validation entities, fold-specific alpha selection, and an all-84 refit only after every development choice is frozen. `scripts/prepare_matr_decoder_crossfit_20260828.py` builds a target-blind fold manifest from label/cycle/protocol columns only, verifies the exact Batch1+Batch2 metadata union and hashes, and forbids Batch3. The authoritative generated manifest is `data/matr_cross_batch_20260828/decoder_crossfit_manifest.json`: 84 entities, validation fold sizes 16/17/17/17/17, seed 20260828, exact one-fold validation coverage, and no target read during assignment. Its focused test passes.

A fresh Luna audit found that the first Batch2 analyzer draft was not yet reviewer-safe: it did not prove exact 43-cell query coverage, launcher terminal states, the full 47-candidate CPU selection, C150 source values, or actual delete-10 reliability, and its Batch3 protection depended too much on path names. Do not execute or trust that draft's gate. A repair is active to cross-check every artifact against `batch2.csv`, manifests, ledgers, hashes, frozen grids, and the new jackknife output; the analysis directory remains absent.

## 61. First finite repaired q results, provenance repair, and symbolic protocol hardening (2026-08-28 17:02 CST)

The empty-prefix repair is scientifically validated: seeds 0--3 in `neural_epoch_normalized_nonempty` completed with finite Batch2 predictions. Their reference/macro per-cell NRMSE pairs are seed0 `1.53447/1.28348`, seed1 `1.28927/1.05877`, seed2 `1.56033/1.31777`, and seed3 `1.32696/1.07045`. Every run has exactly 148,000 theta steps; q steps are 147,998--147,999 with one or two explicitly skipped empty q phases; the global regularizer scale sums to exactly one per epoch. This proves that the prior failures were implementation defects, not evidence that the latent method cannot fit MATR. Seed4 is still active on GPU5. Foreign VLLM jobs occupy GPUs2--4; do not touch them.

The current root is retained as development/diagnostic evidence but is not the formal five-seed root. During seed0--2 result serialization, a future Batch3 symbolic-gate paragraph was temporarily present in the cross-batch plan, so those three results record plan hash `03db20ff...`, while seed3 and the restored plan use `43e61c9b...`. The paragraph did not alter code, training settings, Batch2 gates, or any q outcome, but mixed provenance is below the intended paper standard. Existing results were not edited. The waiting prior session was stopped before it launched any job. The mixed root reached 5/5 terminal success at 17:00 CST. `matr_dev_neural_clean_20260828` then started a fresh five-seed root at `neural_epoch_normalized_nonempty_clean` under plan/runner/pipeline hashes `43e61c9b...` / `bd9dda18...` / `3510bc1a...`; because foreign VLLM jobs still occupy GPUs2--4, only seed0 is active on GPU5 and the other four jobs remain guarded. `matr_dev_latent_prior_clean_20260828` waits behind that clean root and will write `latent_prior_nonempty_clean`. Do not modify `MATR_CROSS_BATCH_CONFIRMATION_PLAN_20260828.md`, `scripts/run_iclr_real_discovery.py`, or `lvs/core/pipeline.py` until the clean q root is terminal.

The decoder-cross-fit symbolic protocol received a strict pre-execution repair after a fresh review. Formula fitting now uses exactly 16 deterministic post-support rows per training battery, validation uses query rows only, `Z100` is diagnostic-only and forbidden from primary formulas, and the primary functional interface is `Z150/Z200/Z300`. Fold-local equations evaluate the discovery procedure; pooled out-of-fold development scores select an algorithm/interface family, never an attractive fold equation. One final expression is generated only by the locked all-84 refit. Reused five-fold numbers are explicitly development-selection scores; untouched Batch3 is the only confirmation. Alpha calibration inherits the exact four-start/200-step/selection/refinement protocol. The all-data refit/hash order is now unambiguous, and Batch3 symbolic incremental-value and motif thresholds are frozen directly in the symbolic protocol without changing the plan file while q jobs are live.

The generated cross-fit manifest now hashes `source_action_ledger.csv` and `cell_audit.csv` as well as the three prepared tables, protocol, and target-blind manifest builder. The fold runner checks every data/protocol/builder/audit hash, exact fold CSV paths, target-blind flags, label isolation, formal 1,000-epoch settings, theta/q/backward/example counts, complete q/Z/query coverage, finiteness, and query-target perturbation invariance. A `--smoke` flag is required for shortened non-counted runs. The launcher now writes live and terminal aggregate state, verifies every `result.json` and artifact, never retries, and exits nonzero unless all 25 independent fold/seed jobs succeed. Three focused manifest/runner tests pass. Do not launch these 25 jobs unless the strict Batch2 analyzer passes the clean q/prior development gate.

## 62. Strict Batch2 analyzer is implementation-complete (2026-08-28 17:12 CST)

`scripts/analyze_matr_batch2_development_20260828.py` now rejects partial or forged evidence before writing any analysis directory. It checks the exact 43 Batch2 labels and every authoritative `(label,cycle,target)` query row; all old/no-q/corrected-q/attentive/prior launcher terminal records; all 47 CPU candidates and their frozen selection tie break; result/artifact containment and hashes; exact job configurations and counters; C150 recomputed cycles-141--150 targets; and ten delete-10 jackknife blocks per cell/seed. It requires all five q results to share one plan/runner/data identity and requires the q root's launch manifest to pin the current plan, runner, and pipeline hashes. The clean q experiment manifest has those launch hashes; individual generic result payloads do not contain a pipeline hash and are not silently treated as if they did.

The analyzer's ICC(3,1) uses the two-way mixed consistency residual with both subject and deleted-block means removed. Scientific alignment is computed on the per-battery cross-seed median C150, while per-seed correlations remain descriptive. It reports battery-level seed-median prediction metrics, paired cell ratios, fixed-seed bootstrap intervals, tie-aware Wilcoxon/BH results, and delete-10 reliability. Passing all frozen Batch2 gates authorizes the symbolic closed loop but never Batch3 directly. Both `--q-root` and `--prior-root` allow the clean provenance roots without renaming or merging artifacts. Ten adversarial focused tests pass, including missing query rows, fabricated C150, nonterminal launchers, mixed q provenance, clean-root CLI routing, exact delete-10 intervals, source-checkpoint binding, train/test-label isolation, and jackknife pass/fail. The analyzer has not been run on real outputs because clean q/prior are not terminal.

At 17:14 CST clean q seed0 completed successfully and seed1 started on GPU5. Its prediction metrics, test metrics, and every optimization counter are exactly equal to the earlier mixed-provenance seed0 result; only the intended plan hash and wall time differ. This directly confirms that the provenance repair did not alter the scientific computation. Four clean q seeds remain.

## 63. Clean q reaches 3/5 and Stage-1 symbolic execution is reviewer-hardened (2026-08-28 17:48 CST)

The formal clean q root has three terminal successes: seeds 0, 1, and 2. For every one of these seeds, deleting only provenance/path/wall-time fields gives an exact zero diff against the preserved mixed-provenance result. Seed3 is active on GPU5 and seed4 remains queued. Host GPUs2--4 continue to carry foreign VLLM loads of about 64.5 GiB each, so the guard is correctly keeping the campaign serial on GPU5. The clean fixed-prior watcher has not started a prior cell and still waits for all five q seeds.

The strict Batch2 analyzer passed all ten tests after its final audit repairs. It now additionally verifies exact delete-10 intervals, true cross-seed C150 ICC separately from delete-block reliability, canonical q checkpoint/result binding for every prior seed, prior-artifact containment, and exact disjoint 41/43 train/test labels. The combined focused Batch2/crossfit/symbolic verification is 18 tests passing. Do not execute the real Batch2 analyzer until both clean q and clean prior reach terminal success.

The first symbolic discovery implementation is now ready but remains gated. `scripts/run_matr_first_symbolic_discovery_20260828.py` and its analyzer consume all 25 decoder-cross-fit results and compare condition-only, support-summary, free functional-q PySR, and a predeclared phase-scaffold functional-q family. Formal PySR is explicitly pinned to version 1.5.10, 80 iterations, maxsize 14, `{+,-,*}`, deterministic serial execution, and training-only `model_selection="best"`. A pre-existing exact PySR 1.5.10 environment at `/public/home/wangyg/workspace/llm_pysr_project/.venv/bin/python` imports successfully with Julia 1.12.6, so no unpinned dependency installation is needed.

A Luna leakage audit confirmed that current formula fitting, target/input scaling, phase coefficients, coordinate clipping, and PySR equation selection use formula-training data only; all validation query targets are scoring-only. Its provenance findings were repaired before formal execution: decoder-crossfit artifacts are now restricted to their exact cell roots, every source artifact is SHA-256 bound in both the crossfit runner and downstream symbolic validator, and the analyzer rechecks canonical terminal ledger rows and result hashes. The symbolic analyzer independently recomputes train-only clipping, input/target scalers, physical-expression predictions, battery residual metrics, and per-coordinate first and time-mixed derivatives. Five symbolic tests, including validation-target perturbation and artifact-escape attacks, pass; together with decoder-crossfit tests the current focused set is eight passing. No formal decoder cross-fit or symbolic cell has been launched, and Batch3 targets remain sealed.

One real non-counted PySR integration smoke was then run on a temporary 10-battery fixture using the exact external PySR 1.5.10 environment. The first attempted smoke failed visibly before a cell because PySR requires `maxsize>=7`; formal `maxsize=14` was unaffected, the failed temporary root was preserved, and the runner now rejects smaller values before dispatch. A fresh current-code smoke with one fold, one seed, one iteration, and maxsize 7 completed all four families and passed the full analyzer. It remains `scientific_selection_eligible=false` and cannot authorize an all-84 refit. The final current-code temporary root is `runs/_tmp_archive_20260828/matr_symbolic_real_pysr_smoke_v3_F1wJFZ/symbolic` (migrated from `/tmp` on 2026-08-28).

## 64. Clean q is terminal 5/5; clean prior and the guarded Batch2 gate are active (2026-08-28 18:03 CST)

The provenance-clean q root completed all five seeds with five zero-return-code ledger rows. Every clean seed is exactly identical to its preserved mixed-root counterpart after excluding only provenance/path/wall-time fields. The formal q scientific values are therefore unchanged, while all five now share plan/runner/pipeline hashes `43e61c9b...` / `bd9dda18...` / `3510bc1a...`. The freeze on those three files remains in force through clean-prior completion.

`matr_dev_latent_prior_clean_20260828` then launched correctly. Its `launcher_manifest.json` binds the exact five clean q result paths, seeds 0--4, rank 2, weight 0.01, probe cycles 25/50/75/100, 1,000 expected source epochs, and the `<128 MiB` availability guard. At 18:03 CST seed0 is active on GPU5 at about 1.1 GiB. Foreign VLLM jobs still occupy GPUs2--4 near 64.5 GiB each, so seeds1--4 remain guarded; cards0/1/6/7 are also foreign and are not used.

A persistent no-retry gate watcher is active in tmux `matr_batch2_gate_crossfit_20260828`. It waits for the clean-prior tmux to exit, runs the strict Batch2 analyzer with explicit clean q/prior roots, checks `authorize_symbolic_closed_loop == true`, and only then launches the exact 25 decoder-crossfit jobs at `runs/matr_decoder_crossfit_20260828` using GPUs2--5 with the same availability guard. An analyzer error or a failed scientific gate stops the chain before decoder-crossfit. Batch3 remains sealed and is not referenced by this watcher.

## 65. Verified live prior and added the gated Stage-1 continuation (2026-08-28 18:13 CST)

A fresh host-level audit confirmed that clean prior seed0 is genuinely active on GPU5, not merely recorded in the stale handoff snapshot. PID `4138508` is the approved CUDA runner under the clean prior launcher, uses about `1.1 GiB`, and executes the exact seed0 command against `latent_prior_nonempty_clean`. At 18:10 CST it was still computing normally and had not yet emitted a terminal `result.json`; this runner writes terminal artifacts rather than partial scientific outputs. GPUs2--4 remain occupied by foreign VLLM jobs near `64.5 GiB` each, and GPUs0/1/6/7 also carry foreign workloads. No foreign process was touched and the `<128 MiB` guard correctly leaves seeds1--4 queued for GPU5.

The eight key frozen hashes were recomputed and remain unchanged: the Batch2 plan, generic real runner, pipeline, symbolic protocol, decoder-crossfit runner/launcher, and Stage-1 symbolic runner/analyzer all match the hashes recorded above. The combined strict Batch2 analyzer, decoder-crossfit manifest/runner, and first-symbolic tests pass `18/18` under `.venv-lvs-gpu`; the separate CUDA environment lacks pytest, which is an environment fact rather than a test failure.

A second persistent no-retry watcher is now active in tmux `matr_stage1_symbolic_20260828`. It waits for the existing Batch2-gate/crossfit session to disappear, then requires the decoder-crossfit snapshot to report exactly `25` planned, `25` terminal, `25` successful, zero failed/pending/running jobs. Only after that check, and only if the formal output root is still absent, it runs the frozen 100-cell first symbolic discovery with PySR 1.5.10, runs the independent analyzer, and checks `authorize_locked_all84_first_formula_refit == true`. If the Batch2 gate fails, crossfit is incomplete, any Stage-1 cell fails, or the first-formula scientific gate is false, the chain stops visibly and does not proceed to the all-84 refit or Batch3. The Stage-1 output root is absent while the watcher waits.

## 66. Prior seed0 succeeds and the downstream execution semantics are frozen (2026-08-28 18:32 CST)

Clean prior seed0 completed in `912.79 s`, wrote every expected artifact, returned code zero, reproduced the source weight-0 prediction to maximum absolute difference `5.96e-08`, and had exact zero query-target perturbation difference for q/C100/C150. Its partial seed-level all-query median reference NRMSE is `1.534923` for prefix q and `1.533037` for the rank-2 prior; the paired cell-ratio median is `1.000030` with 17/43 strict prior wins. Its descriptive C150 empirical-alignment Spearman is `0.880852` for prefix q and `0.867865` for the prior. These are one-seed development observations only; they do not decide the five-seed gate. The launcher immediately dispatched clean prior seed1 on GPU5, while seeds2--4 remain guarded.

A Luna read-only audit exposed two execution ambiguities that had to be resolved before any formal decoder-crossfit or Stage-1 symbolic outcome: Stage1 yields one equation per fold and neural seed, and an all-84 decoder must not participate in alpha or second-formula selection. The implementation-level contract is now frozen in `MATR_SYMBOLIC_CLOSED_LOOP_EXECUTION_CONTRACT_20260828.md`, SHA-256 `3ab2a8bd12bbe43c756f1e69a53953aa00dba135341601b40d839920e2a1b0d0`. It changes no scientific threshold or Stage1 family. The structured cell `(fold,seed)` uses its matching equation and decoder; global alpha is selected from battery-level median predictions across all five seeds; only validation batteries are recalibrated for every alpha, after which both training and validation batteries are recalibrated once through the selected alpha for Stage2; the second formula uses coordinates from the structured predictor; and the all-84 decoder/formulas are created only after every five-fold development choice and gate is frozen. The final single expression uses per-row median functional coordinates across the five all-84 decoder seeds, preventing attractive-seed selection. The contract also requires differentiable Torch formula evaluation during q calibration, physical-unit agreement, complete artifact hashing, and a pre-Batch3 package seal.

## 67. Clean prior reaches 2/5 and seed2 is active (2026-08-28 18:46 CST)

Clean prior seed1 completed successfully after `1,443.37 s`; the launcher recorded return code zero and immediately started seed2 on GPU5. Seeds0--1 are the only terminal prior results at this timestamp. Seed1's all-query median reference NRMSE is `1.390209` for prefix q and `1.344509` for the rank-2 prior. Its prior/prefix paired cell-ratio median is `1.000093` with 18/43 strict prior wins, so the lower marginal median error must not be misread as a broad cellwise win. Seed1 C150 empirical-alignment Spearman is `0.945334` for prefix q and `0.962096` for the prior. The seed0/seed1 C150 cross-seed Spearman is `0.872546` for prefix q and `0.870734` for the prior: above the frozen worst-pair floor `0.75` but below the required ten-pair median `0.90`. With only one of ten seed pairs available, the stability gate remains unresolved and no interface or downstream action is selected from this partial observation.

## 68. Project `/tmp` material migration and persistent storage rule (2026-08-28 19:14 CST)

At the user's direction, every top-level `/tmp` entry unambiguously owned by this project (`lvs*` and `matr*`) plus the three verified MATR pytest fixture roots was moved to `runs/_tmp_archive_20260828/`. A content-evidence scan then found and moved three additional project files without those prefixes: two decoder-crossfit manifest snapshots and one canonical symbolic dry-run configuration. The migrated payload contains 1,194 files and occupies 40,159,797 bytes; the archive contains 1,195 files after adding its migration manifest. The original named sources are absent after the move. The final manifest snapshot is byte-identical to the authoritative repository manifest (SHA-256 `e94f5b6a...`). Unknown `/tmp` files and other projects were not touched.

Durable rule: do not place this project's experiment outputs, smoke roots, pytest base directories, logs, or caches in `/tmp`. Use `runs/` for experiment/test artifacts and `runs/_runtime_cache/` for Matplotlib/XDG caches. All unsealed hard-coded cache defaults and current command examples were updated accordingly. One deliberate temporary code exception remains: `scripts/run_iclr_real_discovery.py` retains its prior `/tmp` fallback until the active provenance-clean q/prior chain is terminal because its existing outputs are bound to that exact runner hash. The already-running chain predates this storage rule; do not launch any new job through the fallback. After the chain is terminal and its strict analysis is reconciled, migrate those two defaults in a separately recorded provenance transition.

The path migration did not interrupt the live experiment. At 19:19 CST clean prior seeds0--3 had four zero-return-code ledger rows and four `result.json` files; seed4 was active on GPU5 at about 1.1 GiB. No project `lvs*` or `matr*` path had reappeared under `/tmp`. The updated decoder-crossfit path default passed the focused manifest/runner/first-symbolic suite (`8 passed`), all 19 touched Python files passed bytecode compilation with `PYTHONPYCACHEPREFIX` under `runs/_runtime_cache/`, and `git diff --check` passed.

## 69. Clean prior terminal, clean no-q provenance repair active, and interpretability gate amended (2026-08-28 20:03 CST)

The clean rank-2 prior root reached 5/5 terminal success at 19:31:59 CST with five zero-return-code ledger rows. The first strict Batch2 analysis attempt then failed before writing an analysis directory because the historical no-q `result.json` payloads predate the now-explicit `normalize_global_regularizers_per_epoch=false` field. The exact failure was a frozen-protocol mismatch (`observed: None`, `expected: False`), not a scientific gate failure. The analyzer was not weakened to infer a missing field. Instead, a provenance-clean no-q launcher was added and focused tests passed.

At 20:00 CST host-level `nvidia-smi` showed GPU5 empty at 4 MiB while GPUs2--4 carried foreign VLLM workers near 64.5 GiB. Tmux `matr_noq_clean_20260828` started the five fixed no-q seeds under the current plan/runner/pipeline hashes, 1,000 epochs, first-100 support prefix, and explicit epoch-normalization false. It writes to `runs/matr_cross_batch_development_20260828/no_q_mlp_clean`, uses only project-local runtime caches, has no automatic retry, and dispatches only below 128 MiB. At 20:02 CST seed0 was genuinely active on physical GPU5 at about 817 MiB; no result was yet terminal.

At 20:19 CST clean no-q seed0 completed successfully in 1,102.70 seconds and seed1 started on GPU5. The formal seed0 reference/macro NRMSE is `1.6541845/1.4042072`, with exactly 148,000 theta/backward steps, zero q steps, and 37,693,000 processed examples. Its prediction dictionary, non-trace optimization counters, and query-prediction CSV are byte/numerically identical to the historical seed0 result; the intended difference is that the new job explicitly records `normalize_global_regularizers_per_epoch=false` instead of omitting the field. This confirms the clean rerun repairs provenance without changing the scientific baseline.

A first pair of shell-only waiting watchers was stopped before either left its wait loop because inspection found that their nested Python gate-check quotes would have been split incorrectly. They produced no analysis or experiment artifact and did not interrupt clean no-q. They were replaced by one source-auditable Python controller, `scripts/watch_matr_clean_to_stage1_20260828.py`, in tmux `matr_clean_to_stage1_20260828`. Its live status is `runs/matr_clean_to_stage1_watcher_20260828/status.json`. It waits for clean no-q, refuses to reuse any downstream output root, runs the strict Batch2 analyzer with explicit clean no-q/q/prior roots, launches the 25 decoder-crossfit cells only if `authorize_symbolic_closed_loop` is exactly true, requires exact 25/25 crossfit success, then runs/analyzes the formal 100-cell PySR Stage 1 and stops unless `authorize_locked_all84_first_formula_refit` is exactly true. It is launched through `.venv-lvs-gpu` with the externally supplied Julia project binding below and uses only project-local caches.

The user explicitly clarified that an interpretable expression need not recover the original raw q gauge. `MATR_SYMBOLIC_INTERPRETABILITY_AMENDMENT_20260828.md` was frozen before formal crossfit/symbolic results. It defines a separate stage-wise interpretability lead: entity-held-out/OOF pooled physical-unit R2 at least 0.85, median prediction across all five neural seeds, no query-target leakage, a post-support functional coordinate in at least 4/5 folds for every seed, finite predictions, and no battery worse than ten times the support-summary NRMSE. The analyzer also writes per-battery R2, median battery R2, the fraction of batteries at or above 0.85, and the existing NRMSE summaries. The original strict predictive gates remain unchanged and independently visible; either the strict gate or this interpretability-lead gate can authorize continued structure/re-q investigation, but neither authorizes Batch3 by itself.

For the later CUDA + PySR closed-loop stage, `.venv-lvs-gpu` now contains exact PySR 1.5.10 and retains Torch 2.11.0+cu128 with CUDA available. It was validated without a segfault by binding to the existing Julia installation/project through `PYTHON_JULIAPKG_EXE=/public/home/wangyg/workspace/llm_pysr_project/.venv/julia_env/pyjuliapkg/install/bin/julia`, `PYTHON_JULIAPKG_PROJECT=/public/home/wangyg/workspace/llm_pysr_project/.venv/julia_env`, and `PYTHON_JULIAPKG_OFFLINE=yes`; the focused closed-loop suite passed 7 tests. Every formal closed-loop command must include those three environment variables. They are intentionally external rather than hard-coded so the repository remains portable.

## 70. Two-channel symbolic gate and full pre-Batch3 execution chain (2026-08-28 20:28 CST)

The interpretability amendment is now propagated through the entire development loop rather than only Stage 1. Stage 1 separately records its overall predictive winner and `selected_functional_family`; B/C separately records its overall second-family winner and `selected_second_functional_family`. This prevents a support-summary predictive winner from hiding a functional formula whose entity-held-out seed-median OOF R2 reaches the explanatory threshold.

B/C now writes pooled and per-battery entity-OOF R2 for every second-stage family after pointwise median prediction across all five neural seeds. Its strict predictive gate is unchanged. A separate second-symbolic interpretability lead requires formal integrity, exact query-target symbolic-input invariance, a post-support functional coordinate in at least four of five folds for every seed, finite predictions with no >10x support-summary battery failure, and pooled OOF R2 at least 0.85. It deliberately does **not** require the structured predictive gate. `authorize_all84_refit` is the explicit OR of strict-predictive success and this interpretability lead. A focused regression proves `strict_predictive=false, interpretable_second_lead=true` still authorizes a formal non-smoke all-84 refit.

The new all-84 runner/analyzer trains five all-84 source decoders, support-calibrates q for every Batch1+2 entity with leakage twins, fits the first formula from pointwise median functional coordinates across exactly five seeds, applies the frozen differentiable structured decoder and alpha, recalibrates q, fits a second formula from the second five-seed median coordinates, independently reproduces source/structured/formula numerics, hashes every upstream and package file, and seals the package before any Batch3 access. It consumes the functional-family choices, never the possibly nonfunctional overall winners. The seal permits only a later single Batch3 evaluation and forbids any refit after that point.

Main-agent verification of the final new gate code is stronger than the initial subagent report: B/C plus all-84 focused tests passed `18/18`, and the combined decoder-crossfit + Stage1 + B/C + all-84 suite passed `26/26`. All touched scripts passed bytecode compilation, `git diff --check`, and scans found no `/tmp`, `batch3.csv`, or `batch4.csv` reference. Formal PySR/Julia direct-CLI execution remains pending its real gated inputs; ordinary module imports no longer initialize or mutate the external Julia project, while formal non-dry-run CLIs require JuliaCall to initialize before Torch.

A second source-auditable controller, `scripts/watch_matr_stage1_to_all84_20260828.py`, is active in tmux `matr_stage1_to_all84_20260828`; its status is `runs/matr_stage1_to_all84_watcher_20260828/status.json`. It waits for the first controller to report `stage1_authorized`, refuses to reuse B/C or all-84 roots, waits for a GPU among 2--5 below 128 MiB before each GPU stage, runs/analyzes B/C, continues only through the predictive-or-interpretability OR gate, runs/analyzes the all-84 package, and stops after verifying the pre-Batch3 seal. It does not run Batch3.

## 71. Pre-Batch3 package audit and explicit two-track authority (2026-08-28 20:41 CST)

A fresh independent read-only audit found that the implemented all-84 seal was still insufficient for legal Batch3 access. It lacked all-84 condition-only and support-summary symbolic comparators, the frozen dominant-motif replacement effect/sign, and the declared no-q/attentive/RF/FPCA baseline refits. It also had no joint symbolic+baseline seal or single-use Batch3 evaluator. Therefore the earlier all-84 analyzer behavior that could set `authorize_unique_batch3_evaluation=true` for any formal non-smoke symbolic package is invalid and is being removed before any formal all-84 result exists. Batch3 remains unread.

The user's R2 clarification is now made protocol-explicit in `MATR_SYMBOLIC_INTERPRETABILITY_AMENDMENT_20260828.md`, still before any formal decoder-crossfit or symbolic result. There are two independent tracks:

- the predictive track retains every original Batch2/Batch3 superiority, stability, alignment, subgroup, and integrity gate;
- the symbolic track may continue from Stage1 through B/C and all-84 when its five-seed entity-OOF R2 is at least 0.85 with zero leakage, functional-coordinate recurrence, finiteness, and no >10x battery failure, even if a strict predictive gate fails.

The symbolic track can authorize exactly one later **symbolic-track** Batch3 evaluation only after symbolic comparators, dominant motif, all predictive baselines, data audit, code, environments, and both packages are in one verified joint seal. It cannot authorize refitting, exclusions, threshold changes, or a predictive-superiority claim. A Batch3 symbolic-track pass still requires all locked second-vs-condition, second-vs-first, structured-vs-source, motif-effect/sign, finiteness, and robustness gates; failure ends the real-data closed-loop claim.

At 20:37 CST clean no-q reached 2/5. Seeds0--1 are exact matches to their historical prediction dictionaries, non-trace optimization counters, and query-prediction CSVs; only the formerly missing explicit protocol field is repaired. Seed2 is active on GPU5. Three implementation/audit tasks are active in parallel: completing symbolic comparators/motif seal and removing premature Batch3 authorization; building an all-84 baseline refit/analyzer package; and auditing every remaining pre-Batch3 gate/artifact. The downstream watcher must be revised to stop at `symbolic_package_pending_baselines` until the joint seal exists.

## 71. Independent all-84 baseline refit package implementation (2026-08-28)

Added the separate pre-confirmation baseline package runner/analyzer/test suite: `scripts/run_matr_all84_baseline_refit_20260828.py`, `scripts/analyze_matr_all84_baseline_refit_20260828.py`, and `tests/test_matr_all84_baseline_refit_20260828.py`. The runner consumes only the frozen development `batch12.csv` and Batch1->Batch2 CPU selection, refits no-q MLP and attentive CNP for seeds 0--4, refits the selected Random Forest, and stores an all-84 train-only FPCA basis/mean/imputation state. Persistence, Huber, and kNN are emitted as selected support-only configurations with `all84_targets_used_for_fit=false`, not refit against all-84 targets. Every model/result has artifact hashes, an explicit fit partition, support boundary, a target-perturbation interface check, and a later-prediction loadable state; package inventory and a pre-confirmation seal detect tampering or unlisted files. No formal run was launched in this subtask; the seven focused tests pass. The initial smoke was stopped before training after exposing a stale historical CPU-selection plan hash and a public-cache issue; the runner now preserves both the historical selection hash and current plan hash and defaults caches to `runs/_runtime_cache/`.
The baseline manifest/results/analyzer also bind `MATR_SYMBOLIC_INTERPRETABILITY_AMENDMENT_20260828.md` by SHA-256. Its audit emits only `authorize_joint_pre_batch3_package_assembly=true` (plus `baseline_refit_package_assembled=true`) and keeps `authorize_unique_confirmation_evaluation=false`; the baseline package never directly authorizes the later confirmation evaluation.

## 72. Symbolic all-84 component seal completed; Batch3 remains unauthorized (2026-08-28 20:52 CST)

The all-84 symbolic runner/analyzer now close the previously identified symbolic-package gaps without reading or running Batch3. In addition to the locked first and second functional expressions, the runner fits and hashes one all-84 `pysr_condition` condition-only comparator and one all-84 `pysr_support_summary` comparator. Their prediction artifacts expose only their declared condition/support inputs and the independent analyzer reconstructs first-100 support summaries, expression predictions, scalers, row selection, artifacts, and hashes. The symbolic package remains separate from the baseline package.

The dominant development motif is now selected from the frozen second functional family by the protocol rule: for each of `Z150`, `Z200`, and `Z300`, replace the coordinate by its Batch1+2 population median after per-battery five-seed coordinate aggregation; take the median absolute prediction change over query rows, then seeds, then batteries; choose the largest effect with lexical coordinate order as the tie break. The exact motif identity, replacement value, battery/coordinate evidence tables, median effect, and sign of the median `d²g/(dt dZ)` are frozen and hashed. `Z100` is explicitly ineligible. The analyzer independently recomputes these quantities from the 5x5 entity-held-out development prediction artifacts rather than trusting the runner's decision file.

The interpretability amendment path/hash is now bound through B/C manifests, alpha and second-formula result source hashes, all-84 manifests/results/inventory/seal, and both independent analyzers. B/C exposes unambiguous two-track fields: `authorize_interpretability_research`, `authorize_all84_symbolic_refit`, `authorize_symbolic_track_batch3_evaluation`, `strict_predictive_gate_passed`, and `strict_predictive_track_batch3_passed`; the legacy `authorize_all84_refit` is retained only as an equality-checked compatibility alias. Entity-held-out seed-median OOF `R² >= 0.85` may advance the symbolic research/all-84 track under the other frozen interpretability conditions, but `interpretability_r2_replaces_predictive_gates=false` and no predictive-superiority conclusion is inferred.

The all-84 analyzer now emits only `authorize_baseline_refit_package_assembly=true` for a formal verified symbolic package. It records `symbolic_package_seal_verified=true`, while `baseline_refit_package_assembled=false`, `complete_pre_batch3_package_sealed=false`, `authorize_symbolic_track_batch3_evaluation=false`, and `authorize_unique_batch3_evaluation=false`. The Stage1-to-all84 watcher requires exactly those states, writes terminal state `symbolic_package_pending_baselines`, and stops normally; it does not invent or invoke the later baseline/joint-seal interface.

Focused B/C plus all-84 verification passes `19/19`; all five touched scripts compile and `git diff --check` passes. A broader `tests/test_matr_*20260828.py` run passed `57` tests and had one unrelated pre-existing Batch2 analyzer-test failure caused by the test itself calling `tmp_path.relative_to(PROJECT_ROOT)` for a pytest root outside the repository. A static scan found no `read_csv`, `read_text`, `_read_json`, or `Path.open` call targeting Batch3/Batch4 and no true Batch3 authorization assignment. The amendment was previously hidden by the root-Markdown ignore rule, so `.gitignore` now explicitly permits `MATR_SYMBOLIC_INTERPRETABILITY_AMENDMENT_20260828.md` to be tracked on the research branch. No formal symbolic/all-84/baseline experiment was launched in this implementation task.

## 73. R2 success criterion finalized; Batch2 research entry decoupled; clean attentive v2 active (2026-08-28 21:09 CST)

The user reconfirmed that the mandatory symbolic endpoint is an interpretable, stage-wise scientifically suggestive expression with entity-held-out/OOF physical-unit `R² >= 0.85`; it need not recover the original raw-q gauge or the originally imagined physical variable. The current amendment SHA-256 is `e9a5f904279d5fec8e183b3414ab256c3660ab0fd1cd603d60684d9e6d2c2d2e`, still frozen before any formal decoder-crossfit or symbolic result.

The Batch2 analyzer previously overloaded its strict predictive gate as the only permission to start symbolic discovery. This was repaired without weakening the predictive claim. It now exposes `strict_predictive_gate_passed` separately and authorizes development-only `authorize_interpretability_research` from complete provenance/integrity/zero-leakage evidence. Pre-symbolic C150 stability, alignment, and delete-support reliability remain visible diagnostics and strict-predictive gates, but failure of one particular coordinate cannot prevent testing structure-recalibrated coordinates. `authorize_symbolic_closed_loop` is the explicit OR of the strict predictive gate and this research-only entry. A targeted counterexample makes no-q prediction better and still requires the symbolic research path to remain open; a second counterexample fails C150 jackknife reliability and likewise preserves research authorization. Neither case authorizes all-84 or Batch3. The Batch2 suite passes `13/13`; after adding the clean-attentive root interface it passes `14/14`. The combined Batch2 + B/C + all-84 suite passes `32/32`.

The provenance-clean no-q root reached 5/5 terminal success. The first automatic Batch2 analyzer then failed before creating its analysis root because the completed attentive-CNP results carry historical plan hash `0147942e...` rather than the current frozen plan hash `43e61c9b...`. This is a provenance failure, not a scientific score failure. The analyzer was not relaxed and old results were not modified.

The first clean attentive retry root `attentive_cnp_clean` is preserved as an infrastructure failure: all five jobs exited before training with `ModuleNotFoundError: lvs`, because direct script execution did not add the repository root to `sys.path`. The runner received the minimal portability repair, direct CLI import under the CUDA environment passed, and six focused attentive tests passed. No scientific output exists in the failed root. A deliberate new-root run is active in tmux `matr_attentive_clean_v2_20260828` at `runs/matr_cross_batch_development_20260828/attentive_cnp_clean_v2`; at launch GPUs2--5 were 4 MiB/0%, seeds0--3 were dispatched one per card, and seed4 remained queued. The first watcher is active and waits for this exact session before using `--attentive-root .../attentive_cnp_clean_v2`; the second watcher waits behind the first. Both use the frozen external Python/PySR/Julia environment and local runtime caches. Batch3 remains sealed and unread.

## 74. Batch2 strictly passes; decoder cross-fit is ready but GPU device mapping is unavailable (2026-08-28 21:31 CST)

The provenance-clean attentive v2 root and CPU baseline root are terminal. The formal Batch2 analyzer completed against the five explicit clean roots and passed every strict predictive gate, not only the research-entry gate. It selected `prefix_q_rank2_prior`, with median per-cell NRMSE `1.4776594310`. Relative to that selected method, the no-q cellwise ratio median is `0.8429062434` with `97.67%` of cells inside the frozen ten-percent rule; the support-kNN overall and late-life ratio medians are `0.8899530445` and `0.8957498004`. C150 median/worst pairwise seed Spearman values are `0.9122082245/0.8707339172`, cross-seed empirical alignment is `0.9525823014`, cross-seed C150 ICC is `0.8662696110`, and the cross-seed-median delete-10 jackknife ICC is `0.9923472781`. `strict_predictive_gate_passed=true`, `authorize_interpretability_research=true`, and `authorize_symbolic_closed_loop=true`; `authorize_batch3=false` remains unchanged. The authoritative artifacts are under `runs/matr_cross_batch_development_20260828/analysis/`.

The first watcher was minimally repaired to resume from this already-written analysis rather than overwrite or recompute it. Resume requires `batch3_read=false`, exact consistency of the two-track authorization fields, and a fresh SHA-256 match for every input listed in `analysis_manifest.json`; crossfit and Stage1 roots must still be absent. The live seal verified successfully, the focused Batch2/crossfit/Stage1 suite passed `24/24`, bytecode compilation passed, and `git diff --check` passed.

The 25-cell decoder cross-fit has not been launched. At 21:30 CST the current container still exposes the host's eight H100 entries under `/proc/driver/nvidia/gpus`, but `/dev/nvidia*` is absent, `nvidia-smi` cannot communicate with the driver, and Torch reports `cuda_available=false` and zero devices. Local SSH attempts could not authenticate to the host namespace. This is a container/device-mapping blocker rather than evidence that GPUs are occupied. Do not create the formal crossfit root until both occupancy and CUDA usability can be verified. While waiting, continue the CPU-only baseline/joint-seal orchestration and single-use-evaluator implementation without reading Batch3.

Because the clean source chain is now terminal and its strict analysis is reconciled, the deliberate storage exception recorded in section 68 is closed. `scripts/run_iclr_real_discovery.py` changed only its two default Matplotlib/XDG cache paths from `/tmp/lvs-*` to repository-local `runs/_runtime_cache/*`; its new SHA-256 is `1d3c1a70379fc94e994514ffda953200169765c1082d5f9bd706508f37c3bdc6`. Historical q results remain bound to the former runner hash and were not edited. Decoder cross-fit uses its own frozen runner and does not import this generic source runner. The new joint pre-Batch3 assembler was also made portable before any formal package exists: upstream symbolic, baseline, and development-audit roots are stored as project-relative paths. Its seven focused tests pass and explicitly reject absolute root serialization.

The remaining pre-Batch3 orchestration is implemented in `scripts/watch_matr_prebatch3_20260828.py`. It waits for the all-84 symbolic watcher state `symbolic_package_pending_baselines`, requires a guarded GPU only for the locked all-84 neural baseline refits, audits that package, assembles and independently audits the joint seal, writes `joint_analysis.json`, and stops at `authorize_single_use_evaluator_construction=true` while every actual evaluation authorization and `evaluator_run` remain false. Main-agent review plus the baseline/joint/watcher suite passes `19/19`; no formal downstream root was created and no Batch3/Batch4 input path exists in this controller.

The joint seal's two-track eligibility derivation was corrected before any formal joint package existed. A strict-predictive confirmation attempt is eligible when the all-84 symbolic decision records the development `strict_predictive_gate_passed=true` and the baseline refit/joint-assembly package is complete; it no longer circularly requires `strict_predictive_track_batch3_passed=true` before Batch3 is evaluated. Likewise, the non-action state `symbolic_track_batch3_eligible` is derived from the all-84 `authorize_interpretability_research` plus the complete baseline package. Actual `authorize_symbolic_track_batch3_evaluation`, `authorize_unique_batch3_evaluation`, and both post-evaluation pass fields remain false in the pre-Batch3 seal. Nine focused joint-package tests cover the strict and symbolic derivations separately.

The single-use evaluator and independent analyzer are implementation-complete in `scripts/run_matr_single_use_confirmation_evaluator_20260828.py` and `scripts/analyze_matr_single_use_confirmation_20260828.py`. The output/lock root is uniquely derived from the joint-seal SHA-256, so a second output path cannot bypass consumption. An exclusive create, 256-bit nonce, append-only hash-chained receipt, and pre-target-access source/seal/prepared-manifest hashes make both successful and failed attempts consume the one opportunity. CUDA unavailability is a visible post-consumption failure rather than a CPU fallback. Inference redacts every query target, allows only the declared first-100 support fits, loads the existing sealed formats without refit/tuning, and leaves all scientific decisions to the independent analyzer. The analyzer rechecks the prepared query targets against every raw prediction row; aggregates seeds by pointwise median and batteries as the statistical unit; computes physical-unit pooled/per-battery R2, 10,000-replicate fixed-seed bootstrap intervals, strict no-q/kNN/stability/alignment/subgroup gates, second-vs-condition/first, structured-vs-source, motif effect/sign, and the no-10x-failure condition. The combined clean-watcher/pre-Batch3/joint/evaluator suite passes `24/24`, both scripts compile, and `git diff --check` passes. No real Batch3/Batch4 target path was read, listed, globbed, or executed.

The final `/tmp` storage audit found only two legacy project cache directories, `/tmp/lvs-xdg-cache` (392 KiB) and `/tmp/lvs-matplotlib-cache` (112 KiB). They contained only fontconfig/Matplotlib cache files and were moved recoverably to `runs/_runtime_cache/legacy_tmp_migration_20260828/`; no project `lvs`, `matr`, or symbolic path remains under `/tmp`. The remaining `/tmp/taylor-svg-render/...` match is unrelated to this project and was not touched.

## 75. Mandatory expression endpoint separated from predictive superiority (2026-08-28 22:04 CST)

The user again fixed the scientific success criterion: finding a comparatively interpretable, stage-wise suggestive expression is sufficient when its five-seed-median entity-held-out/OOF prediction in physical units reaches pooled `R² >= 0.85`. The expression need not recover the initial raw `q`, the originally imagined physical variable, or a unique true law. A decoder-functional or structure-recalibrated coordinate is valid when it provides a recurring relation with scientific heuristic value.

This was already frozen before formal symbolic outcomes in `MATR_SYMBOLIC_INTERPRETABILITY_AMENDMENT_20260828.md`, so that file and its bound SHA-256 were not changed. `ICLR_CLAIM_EVIDENCE_GATE_20260828.md` now makes the distinction reviewer-explicit: the mandatory expression endpoint is independent of the stronger claim that latent q beats no-q MLP, kNN, or other support-aware baselines. Predictive superiority still requires its own frozen gates; a predictive-track failure cannot erase a passing interpretable expression, and an expression pass cannot be mislabeled as predictive superiority.

## 76. Locked-confirmation preparer closes the final evaluator input gap (2026-08-28 22:13 CST)

The pre-confirmation chain previously ended with a verified joint seal and an implemented single-use evaluator, but no formal program generated the evaluator's required prepared manifest/data audit. `scripts/prepare_matr_locked_confirmation_20260828.py` now closes that interface. It first independently validates the joint pre-Batch3 package, fixes its output root by the joint-seal SHA-256, and only then accepts the frozen confirmation source, Batch1+2 identity table, cell audit, and source-action ledger. It checks exactly 40 distinct cells, first-100-plus-20 coverage, strict cycles, constant parseable protocol, finite values, zero label or exact-curve overlap with development, and only the three source-documented Batch3 exclusion families. It computes no model prediction, metric, correlation, normalization, refit, tuning, or formula choice and does not access Batch4.

The preparer emits the exact `matr_locked_confirmation_prepared` and `matr_locked_confirmation_data_audit` contracts required by the single-use evaluator, including hashes for every source, the joint seal, preparer code, prepared table, and audit. Three synthetic-fixture tests prove evaluator compatibility, that a bad joint seal is rejected before any source read, and that an unknown/post-hoc exclusion reason fails. The combined preparer/evaluator suite passes `11/11`; the complete `tests/test_matr_*20260828.py` matrix passes `88/88` in 155.08 seconds; bytecode compilation and `git diff --check` pass. No real Batch3 or Batch4 input was read, listed, globbed, or executed while implementing or testing this component.

At the same timestamp the formal decoder-crossfit and every later symbolic root remain absent. The container still lacks `/dev/nvidia0`, `/dev/nvidiactl`, and `/dev/nvidia-uvm`; `nvidia-smi` cannot reach the driver and Torch reports CUDA unavailable with zero devices. No Slurm command is installed, and Docker/Podman cannot access a usable daemon/runtime. This remains an infrastructure/device-mapping blocker, not a GPU-occupancy observation. Do not create the formal 25-cell crossfit root until CUDA usability and occupancy can both be verified.

`COMPLETE_RESEARCH_REPORT_20260809.md` now has a reader-facing MATR section with a 41/43/40 cohort-role table, the complete nine-method Batch2 table, paired q comparisons, C150 reliability/stability evidence, the remaining 25-cell symbolic path, and the independent `R² >= 0.85` expression criterion. It explicitly reports FPCA as the strongest Batch2 predictor and does not let that predictive result erase the separate symbolic hypothesis. A scan of the main report, README, ICLR gate, real-q milestone, training-dynamics document, and current MATR/NASA plans found no C-MAPSS/turbofan/RUL mention.

## 77. Batch2 source-decoder symbolic bridge active while CUDA is unavailable (2026-08-28 22:44 CST)

Because the formal 25-cell Batch1+2 decoder cross-fit remains blocked by absent CUDA device nodes, a separate development-only CPU bridge was frozen before outcomes in `MATR_BATCH2_SOURCE_DECODER_SYMBOLIC_BRIDGE_PLAN_20260828.md` (SHA-256 `36b67a3c389b79b091fcfa8c04118da83f59f8ed51756adae66b72652c2dde4c`). It uses the five already-completed Batch1-trained source decoders and the five sets of Batch2 first-100-support-calibrated `prefix_q_rank2_prior` q values. On CPU it reconstructs `Z100,Z150,Z200,Z300`, verifies Z100/Z150 against the historical artifacts, and takes the per-battery across-seed median. Raw q is forbidden. The 43 Batch2 entities are assigned target-blind to five folds by sorted-label index modulo five, and every formula-training entity contributes exactly 16 query rows.

The bridge runs the same four families as formal Stage1: PySR condition-only, PySR support-summary, PySR functional-q, and the fixed physical phase scaffold. PySR keeps the frozen `{+,-,*}` grammar, 80 iterations, maxsize 14, deterministic seed `20260828+fold`, and serial within-fit execution. The independent success gate is the user-approved pooled physical-unit entity-OOF `R² >= 0.85` plus zero leakage, post-support coordinate use in at least 4/5 folds, no >10x support-summary battery failure, and a recurring time-by-stage motif in at least 4/5 folds. It is explicitly sequential Batch2 development evidence, cannot replace the formal crossfit, and cannot authorize Batch3.

The JuliaCall project, writable depot, Matplotlib/XDG caches, test scratch, logs, and all experiment outputs are under repository-local `runs/`; no project material uses `/tmp`. A non-counted fold0 four-family smoke passed 4/4 with exact zero query-target symbolic-input difference. The formal runner SHA is `f59c61b454f5f110292b3f0255ec30797824dbd196fef4799c422ed944fcdb2d`; an import-order repair after smoke initializes JuliaCall before Torch to remove the upstream segmentation-fault risk without changing any scientific choice. The analyzer SHA is `5136f2747636394d92609a3bb46bf965e5a62a45648cc1e7512f8a9d9a5867d6`; the final consensus-expression fitter SHA is `5f25402f40bf6f6f8e35cb330ce9bf7df37da5f2e227f857f88b21ce4df1152e`. Five focused tests pass, all three scripts compile, and `git diff --check` passes.

Formal bridge execution is active in unified exec session `21738` at `runs/matr_batch2_source_decoder_symbolic_bridge_20260828`. At 22:44 CST it is 1/20 complete: fold0 `pysr_condition` succeeded in 358.38 seconds and fold0 `pysr_support_summary` has a live, updating hall-of-fame file. Monitor `status.jsonl`; do not inspect partial formulas to change the frozen search. When the manifest reaches terminal success, run the independent analyzer, then the final consensus refit authorized by `analysis/decision.json`. Batch3 remains unused for model evaluation.

At 22:48 CST the formal bridge reached 2/20: fold0 condition-only and support-summary PySR succeeded in 358.38 and 338.52 seconds, and fold0 functional-q PySR is active. Unified watcher session `25553` runs `scripts/watch_matr_batch2_source_decoder_symbolic_bridge_20260828.py` (SHA-256 `0c0ab82e20e261a3820e8e8c34e3c844d237affc80aeae799ebfef861c93eb15`). Its repository-local status is `runs/matr_batch2_source_decoder_symbolic_bridge_watcher_20260828/status.json`. It waits without a deadline, then invokes the already-frozen analyzer and one final consensus refit exactly once; it does not retry failed cells, change any search choice, or touch Batch3.

## 78. Batch2 bridge terminal and seed-local q gauge diagnosis (2026-08-29 00:16 CST)

The CPU source-decoder bridge is terminal at 20/20 successful cells with zero query-target symbolic-input leakage. Its selected `phase_scaffold_functional_q` family has entity-OOF pooled physical-unit `R²=0.5255536842`, battery-bootstrap 95% interval `[0.2281,0.7175]`, post-support use in 5/5 folds, recurring `t:Z150`, `t:Z200`, and `t:Z300` motifs, and maximum per-battery NRMSE ratio 2.6083 versus support summaries. Every frozen gate except `R² >= 0.85` passed, so this is a readable but sub-threshold intermediate relation and does not authorize Batch3. The all-Batch2 consensus refit is explicitly unscored independently; its physical phase expression is preserved under `runs/matr_batch2_source_decoder_symbolic_bridge_20260828/final_consensus_refit/` and must not be presented as held-out success.

The first failure diagnostic shows why extending the same early response grid is unlikely to close the gap. The nine-horizon decoder response panel has effective rank 3 at 95% explained variance and strongly preserves early capacity (`Z100` versus empirical C150 Spearman 0.9689), but entity-OOF ridge prediction from the response grid is negative for knee (`R²=-0.7430`), observed record length (`-0.7458`), and weak for query slope (`-0.0605`). This diagnoses an information-interface mismatch: same-protocol early decoder coordinates preserve early capacity while discarding degradation-stage/lifetime information. It does not diagnose a general failure of latent variables or of the dataset.

The separately frozen Batch2 supervised q-gauge diagnostic is bound to plan SHA-256 `cca49a1275e2af20fc013b606c4c03bf6328f41642c0bc5e1046c9d161b41496` and runner SHA-256 `27fe448f5de122195e42889b49b3baf6f0f239dba0ca56dcfd9a2d6f028eb766`. Seed-local q was mapped within each training fold to physical lifetime, diagnostic knee, and query slope, then physical predictions were aggregated across the five gauges; raw q was never a final-formula input and Batch3 remained unopened. The best q-to-lifetime model reached only OOF `R²=0.2234`, Spearman `-0.0539`, versus support-summary lifetime `R²=0.1945`, Spearman `0.1297`; therefore q does not add reliable held-out lifetime rank information in this interface. Cross-seed physical prediction stability is also only moderate (median pairwise Spearman about 0.41--0.55). Do not relabel this raw q as lifetime or knee.

One constructive lead remains: the support-summary ridge predicts query robust slope with entity-OOF `R²=0.8183` (Spearman 0.4864). The next development experiment should therefore be frozen around a physically readable anchor-plus-slope/curvature structure and compare support-derived versus q-plus-support-derived physical coefficients. Its final curve expression may use a newly recalibrated physical coefficient q and need not recover the original q gauge, but the mandatory success standard remains entity-OOF physical-unit `R² >= 0.85`; predictive superiority remains a separate claim. Formal decoder crossfit is still blocked because this container has no `/dev/nvidia*`, `nvidia-smi` fails, and CUDA is unavailable. Batch3 remains sealed.

## 79. Physical-coefficient re-q and rich-support repair remain sub-threshold (2026-08-29 00:28 CST)

The capacity-only physical re-q plan was frozen at SHA-256 `dd74afbdea92364b978f6be03ea063dbaf2c5fa4a5ebbe9d5e947fb162d363e3`; its runner SHA-256 is `01b7e74d60e74847a7a267ed22394e92d5f68c0b2406945802321dc238604f18`. It replaces the original arbitrary gauge by readable physical coefficients in `capacity=C100+a+b*u` or `capacity=C100+a+b*u+c*u^2`, with `u=(cycle-100)/100`. Coefficients were inferred entity-OOF from support or support-plus-seed-local-q, and raw q was not a final symbol. The selected support-ridge linear family reached pooled OOF `R²=0.6666814292`, improving the source-decoder bridge's 0.5256 but failing 0.85. Its maximum battery NRMSE ratio to direct support linear was 4.2033 and the measured held-out-target perturbation effect was exactly zero. Oracle analysis shows the linear family itself is capped at pooled `R²=0.8034`, while per-battery oracle quadratic coefficients reach 0.9641; therefore a linear formula cannot meet the endpoint, while unconstrained quadratic coefficient transfer is the unstable component.

The raw MATR audit exposed a material information restriction in the prepared cross-batch CSV: the official `.mat` files contain `Qdlin`, `discharge_dQdV`, voltage, internal resistance, temperature, charge time, and other per-cycle arrays, but the current table retains only cycle, protocol, and discharge capacity. This restriction is scientifically important rather than cosmetic. The source study reports that capacity-only first-100 features are weak and that `Delta Q_100-10(V)` plus voltage/sensor features provide the early degradation signal. Only raw Batch1/2 field structure was inspected; Batch3 and Batch4 remained unopened.

The sequential rich-support plan was frozen at SHA-256 `891330fa2aa6cc0b599151da4e1ee2220e94cc3779c0e39e18cd1c3b14a8306a`; runner SHA-256 `592ccf6f974689249af48ba0c702bf3e71ee123348f608fa8134fdcad802347e`. It adds only first-100 Batch2 features: log variance, log absolute minimum, and mean of `Delta Q_100-10(V)`, plus mean/slope/cycle100-minus-cycle10 for QCharge, IR, Tavg/Tmax/Tmin, and charge time. The best family became `rich_q_plus_support_ridge_linear`, with pooled OOF `R²=0.6784728190`, a small positive gain over the capacity-only 0.6667. The remaining rich linear families are 0.6438--0.6752; rich quadratic families improve over the capacity-only quadratic failures but remain only 0.4095--0.4336. The best family is finite, exactly invariant to held-out target perturbation, and has maximum battery NRMSE ratio 4.8629, but still fails 0.85. `Delta Q` features correlate with the transferable linear slope (absolute Spearman about 0.50--0.53) but not the unstable quadratic curvature (absolute Spearman below 0.09).

Durable interpretation: the old q should not be renamed as the desired physical variable, but q plus information-complete support contributes a real, modest increment after structure recalibration. The next formula family must model knee/accelerated degradation with lower effective freedom than a free per-entity quadratic coefficient, for example a frozen shared-shape or fixed-hinge phase law with one transferable acceleration amplitude. Do not add more arbitrary support features or tune on Batch3. The user-approved endpoint remains a suggestive, stage-wise physical expression with entity-OOF `R² >= 0.85`, not recovery of the initial q or a unique true law. Batch3 is sealed and no confirmation is authorized.

## 80. MATR lifetime-stage q approaches but does not cross the expression gate (2026-08-29 01:05 CST)

A lifetime-normalized shared physical law was tested after the rich-support coefficient failure: `capacity=C100+shared_shape(s)`, `s=(cycle-100)/(q_L-100)`. The first rich RF `q_L` reached curve pooled OOF `R²=0.8219869`; oracle lifetime with the same family reached 0.9361, localizing the main limitation to support-to-stage inference rather than expression capacity. Alternating stage-scale re-q reached 0.7807 and its predicted-versus-oracle scale Spearman was only 0.202; a second degradation-rate coordinate reached 0.7995 versus the one-coordinate base 0.8220. These results stop arbitrary extra latent dimensions.

The official-style exact 20 first-100-cycle feature set materially improved the same frozen quadratic expression. Its fixed RF lifetime decoder reached lifetime OOF `R²=0.5554714`, MAPE `0.0872374`, and curve pooled OOF `R²=0.8464411125`. It is finite, has exact zero held-out target-input difference, uses the physical stage coordinate, and has maximum battery NRMSE ratio 6.5136 to direct support linear. This is a genuine near miss but fails the exact 0.85 gate; do not round it to success. The exact20 decision SHA-256 is `ef6dd03e92ce05712cef991c8ee9544b47be1b7ee0a19bbca0f73742d2694141`.

Nested lifetime-RF selection improved lifetime R² to about 0.594 but reduced curve R² to 0.827; curve-targeted inner selection reached about 0.833. A frozen recheck of the same q_L showed the original quadratic remains the best of linear/quadratic/fixed hinges. These failed refinements indicate that optimizing lifetime prediction or adding fixed knees does not automatically optimize the symbolic curve endpoint.

## 81. Low-freedom MATR closure attempts fail without weakening the protocol (2026-08-29 01:16 CST)

The final coefficient-measure alignment changed only shared quadratic training weights and reduced pooled R² from 0.8464411 to 0.8398624. A predeclared low-freedom shape extension selected the cycle-100-anchored quadratic at 0.8459699; anchored and unanchored cubic shapes fell to 0.7925 and 0.7840. Formula-guided q calibration reproduced the source raw q within `5.68e-14`; identity remained best at 0.8464411, while a span power and span scale fell to 0.8193 and 0.7849. Therefore the gap is not explained by point weighting, a missing cubic term, or uniform q-scale bias.

The exact20 interpretable structure-re-q test then predicted either a degradation amplitude or curvature coordinate from nested inner-OOF stage targets. Oracle coordinates show real remaining structure (`R²=0.9401` amplitude, `0.9449` curvature), but held-out predicted amplitude reached only 0.5691 and curvature -0.8119; the selected amplitude also violated the no-10x tail gate. The decision SHA-256 is `a13dada62d968225f51dbba973546a549e184af755af2fba17bc1e3fdaa0f1b9`. Durable conclusion: MATR first-100 data reliably expose one main lifetime-stage coordinate but do not identify the residual cell-specific curve shape. Preserve all 43 batteries and the 0.85 threshold; Batch3 remains sealed and unauthorized.

## 82. Reviewer-clean Starry ZT passes the mandatory interpretable expression endpoint (2026-08-29 01:22 CST)

The eligible reviewer-clean StarryData2 ZT cohort was selected because the existing audit finds `0/80` sample-ID entities with mixed property semantics. It matches the support-conditioned shared-function setting better than lifetime forecasting: each unseen material provides temperature-stratified support measurements and query ZT values remain hidden. The plan was frozen at SHA-256 `f4e4d5b015c46ce398a22b984d4a063609142115fde5de680de978e8d96bcc7f`; runner SHA-256 `412d2476372bc1402caad793e2ab70373bf208bcd9e8bf078c7226b0056fbfe1`.

Across five target-free entity folds and 3,879 query rows, the structure-recalibrated expression `ZT(T)=q0+q1*tau+q2*tau^2`, `tau=(T-mu_train)/sigma_train`, reaches pooled physical-unit OOF `R²=0.9806678897` and RMSE `0.0398682`. The three q coordinates mean reference ZT, first-order temperature sensitivity, and curvature. Entity bootstrap 95% R² is `[0.9617285,0.9915932]`; median entity R² is 0.9834367; 87.5% of entities reach individual R² at least 0.85. Maximum entity NRMSE ratio to linear re-q is 1.0998; all predictions are finite; query-target perturbation changes q/predictions by exactly zero. Decision SHA-256 is `744e3429b5c2fd7537f62bb2ed29d000070b435b37ce4a4bac54bfed4b6a0a1a`; manifest SHA-256 is `dea4058ba670eb75443e51b2d6d08c87753f891f9c9c0769c1de6b11f6237209`.

The q distance geometry has Spearman 0.7222 with empirical 21-point response-curve distances. Changing the temperature-stratified support offset gives q-distance stability Spearman 0.9780, 0.9733, and 0.9679. Comparators on identical queries are support kNN 0.990970, linear re-q 0.966746, no-q global quadratic 0.452321, and no-q MLP -0.932574. Thus the mandatory interpretable expression endpoint is passed on real-data development, while predictive superiority is explicitly not inferred because kNN remains stronger. This cohort and a quick screening were previously exposed; report it as development evidence and obtain a new eligible material cohort or frozen external split before calling it independent confirmation. MATR remains the harder forecasting/gauge boundary case.

## 83. Starry ZT temporal confirmation passes; project `/tmp` remains empty of research material (2026-08-29 01:39 CST)

After the development formula was frozen, the official Starrydata 2026-08-29 daily release was stored under `data/external/starrydata_latest_20260829/`. Target-blind selection used only metadata and `x`: creation after the old snapshot cutoff, sample ID absent anywhere in the old snapshot, exactly one strict ZT-temperature curve with at least 20 points, nonempty DOI/composition, and DOI/composition disjointness from development and within confirmation. It selected 30 entities with 30 unique DOIs and 30 unique compositions. The selection code explicitly excludes `y`; the selection manifest was sealed before target access. Plan SHA-256 is `9b3943b7d5662f01d9cafd023fecb73b5346de272f349df77841ee1f57648817`, preparer SHA-256 `495757f27c32c51214e14dac7a61206523f817d9874ccd5a90dcfd54cc45af34`, sealed selection-manifest SHA-256 `91f920e27bf41d9d5caa1aff6010261fa3e3a6ed5c14b93f90ea6eab63fdde9c`, and evaluator SHA-256 `d1773e314598149d0603011236dbb7bfe28d5c695b38a52915dfff2d7b6055a8`. A consumption receipt was written before target access and forbids rerun.

The unchanged expression `ZT(T)=q0+q1*tau+q2*tau^2` was evaluated once on 919 query rows. It reaches pooled `R²=0.9888097989`, RMSE `0.0370305919`, entity-bootstrap 95% interval `[0.9733056077,0.9947077428]`, median entity R² `0.9417389794`, and 20/30 entities at R² at least 0.85. Maximum entity NRMSE ratio to linear re-q is 1.675607; query-target input difference is exactly zero; all six preregistered gates pass. Comparators are support kNN 0.9826780, linear re-q 0.9642656, no-q MLP 0.0955392, and no-q global quadratic -0.0715156. The expression endpoint is therefore independently temporally confirmed across new sample IDs, papers, and compositions.

Do not convert the pooled kNN comparison into a superiority claim. An explicitly post-confirmation paired description finds lower quadratic NRMSE on 16/30 entities, median quadratic/kNN NRMSE ratio 0.9031, but mean paired NRMSE difference `quadratic-knn=+0.0778`, bootstrap interval `[-0.0111,0.1722]`, and two-sided Wilcoxon `p=0.4645`. `predictive_superiority_inferred=false` remains authoritative. One entity has negative quadratic R² and 10/30 are below 0.85; report the pooled, bootstrap, median, fraction, and worst-tail facts together.

At the user's request, `/tmp` was re-audited by project-name patterns, recent content evidence, and repository path references. No new `lvs`, `matr`, `starry`, latent-variable, or ZT-confirmation experiment material is present there. The prior recoverable archive remains at `runs/_tmp_archive_20260828/` (41 MiB), runtime caches are under `runs/_runtime_cache/`, and all new Starry raw data/results are under the repository's `/public/home/wangyg/latent_variable_search/{data,runs}` tree. The only path-name match is an empty Claude runtime task directory, not research material; unrelated users/projects and runtime sockets were not touched. Durable rule remains: never use `/tmp` for project outputs, downloads, fixtures, logs, plots, or caches. Use `runs/`, `data/external/`, and `runs/_runtime_cache/` so another machine/agent can reproduce the work.

The main report section 23, README, ICLR claim gate, and real-q milestone were updated. The former high-risk item “no untouched real expression confirmation” is closed. The main remaining method risk is that the confirmed re-q step is simpler than the full learned neural q loop; next work should connect learned raw q/decoder response functions to the canonical equation coefficients rather than run more polynomial robustness scans.

## 84. Frozen neural-to-canonical ZT bridge launched on repository-local storage (2026-08-29 01:48 CST)

The old 60-train/20-test ZT neural result was audited before adding new code. It has `train_with_q.csv` and `test_with_q.csv` but no checkpoint, so the decoder cannot be functionally probed. A training-entity-only ridge map from its four raw q coordinates to quadratic coefficients was selected by leave-one-entity-out training error; on the old held-out query it gives `R²=-0.3335`, while direct support re-q on the exact same old prefix support gives `R²=0.9677`. This is a diagnostic of raw-q gauge unreadability, not a failure of the confirmed expression.

The follow-up was frozen in `STARRY_ZT_NEURAL_CANONICAL_BRIDGE_PLAN_20260829.md`, SHA-256 `1b7d1d41052694004f05167634359ee377939ac1b6f959672e07d41a8fb32323`. It uses only the 80 development entities and the same five folds and temperature-stratified support as the confirmed expression development. The 30 temporal-confirmation entities are not referenced or reopened. Fifteen cells cover 5 folds x seeds 0--2. Each uses q-dimension 4, hidden `(256,128)`, 1,000 epochs, matched old HSIC/continuity/q-L2 weights, four-start exact-support q calibration, and then compares raw decoder, raw-q-to-coefficient ridge, decoder-functional polynomial projections of degrees 1--4, and final support structure re-q. Every cell saves checkpoint, support/query mask, q/coefficients, query predictions, training history, runtime, and hashes.

The formal runner SHA-256 is `45d98629e99b6fe58435351c44f4e4ebaa79bd5a6a247886b50d63e2ec7ec251`. The launcher actually used for this run had SHA-256 `84a3dc336859d0f8d857797a6c3336f6ec202884b278752e996cfef443afd8bb`; after terminal analysis its only machine-specific `project_root` assignment was replaced by repository-relative resolution for portability, producing current launcher SHA-256 `0bc91addb8f7f6c158cfd8809b85c098686837e8e2400fc6327e13da9e57433d`. This post-run path-only change does not alter the runner or results and must not be confused with the executed-launcher provenance. Two distinct non-counted two-epoch smokes passed. The final smoke has 830 query rows, exact zero query-target input difference, decoder degree-2 projection fidelity `R²=0.999904`, raw decoder/query `R²=0.6962` at only two epochs, functional degree-2 `0.6961`, raw-q ridge formula `0.2830`, and support structure re-q `0.9932`. These smoke scores are structural only and must not be quoted as scientific results.

The formal campaign is active in tmux `starry_zt_bridge_20260829` under `runs/starry_zt_neural_canonical_bridge_20260829/`, with four CPU workers and four Torch threads per worker. At 01:47 CST fold0 seeds0--2 and fold1 seed0 have running manifests; no cell is terminal yet. The container still lacks `/dev/nvidia*`, so this is CPU execution rather than an artificial VRAM cap. All logs, caches, checkpoints, and pycache are repository-local; no project material is written to `/tmp`. The independent analyzer is implemented at `scripts/analyze_starry_zt_neural_canonical_bridge_20260829.py` and must run only after all 15 formal manifests exist.

At 01:53 CST the user reconfirmed the hierarchy of claims: finding an interpretable expression means a compact, scientifically suggestive stage-wise relation with strict entity-held-out/support-query physical-unit `R² >= 0.85`. It does **not** require recovering the initially learned raw `q`, a unique ground-truth law, or a preselected physical variable. A decoder-functional or support/structure-recalibrated coordinate is eligible. Predictive superiority over kNN/no-q methods, recovery of the original latent gauge, and mechanistic truth are separate stronger claims and must neither be silently required for this endpoint nor inferred from it. Under this durable definition, the Starry ZT expression endpoint is already passed and temporally confirmed; the active neural-to-canonical campaign is method-bridge evidence, not a rerun of the expression-existence gate.

The same 01:53 CST status refresh found 5/15 formal bridge cells terminal-successful and four more active. The five partial cells all have exact zero query-target input difference. Their degree-2 decoder-functional query `R²` spans `0.7937--0.9833`, decoder-response projection fidelity spans `0.9935--0.9993`, structure re-q spans `0.9725--0.9932`, and direct raw-q-to-coefficient ridge spans `-8.3399--0.2320`. These are timestamped partial observations only; do not promote them or alter the frozen protocol before the 15-cell aggregate is analyzed.

## 85. Neural-to-canonical ZT bridge terminal; scale-aware one-factor repair launched (2026-08-29 02:05 CST)

The frozen bridge completed 15/15 cells successfully with exact zero query-target input difference. Across all 80 development entities and 3,879 query rows, the three-seed pointwise median results are: raw decoder `R²=0.948354`, raw-q ridge-to-quadratic `-1.907461`, decoder-functional degree 2 `0.944683` with entity-bootstrap `[0.889918,0.975222]`, and support structure re-q `0.980668`. The primary degree-2 decoder response projection has minimum/median fidelity `R²=0.985033/0.994633`. This is strong pooled evidence that decoder-response projection provides a readable equation coordinate where direct raw-q reading fails.

The full frozen bridge decision is nevertheless `false`. Nineteen of 80 entities exceed ten times the structure-re-q NRMSE, with maximum ratio `744.872`; degree-2 functional median entity R² is `0.846748` and 40/80 entities reach individual R² at least 0.85. Several near-zero-ZT materials are genuine absolute-error failures, not exclusions: target standard deviation has Spearman `-0.614885` with functional NRMSE, and 14/20 materials with absolute mean ZT below 0.02 trigger the tail. Functional distance-geometry stability is also lower than raw (`0.738385` versus `0.794298` median), passing only 8/15 pairwise comparisons. A separate readability fact remains: named functional-coordinate Spearman is `0.733283` versus unaligned raw-coordinate `0.014148`, improving in 15/15 pairs. Do not replace the failed distance gate with this coordinate statistic.

The authoritative result note is `runs/starry_zt_neural_canonical_bridge_20260829/NEURAL_CANONICAL_BRIDGE_RESULTS.md`; raw aggregate decision and tables are under its `analysis/`. The correct claim is pooled neural-to-equation readability with a failed tail-safe/full-stability bridge. The Starry expression-existence endpoint remains passed and temporally confirmed because it uses support structure re-q and does not depend on this stronger bridge.

A single-factor repair was frozen before new results in `STARRY_ZT_SCALE_AWARE_NEURAL_BRIDGE_PLAN_20260829.md` (SHA-256 `82655e3a6e6e68776aae9442e9eb47bd7bc99e5e2c677941654cad8055ca1a95`). It retains all entities, folds, seeds, supports, model, 1,000 epochs, and calibration budget, but uses the outer-train-only invertible target coordinate `z=asinh(y/s_y)`, where `s_y` is the median training-entity population target standard deviation. Physical-unit inversion occurs before decoder projection and scoring. Formal runner SHA-256 is `ef5f0bff2324b84942332beedf4c255412fa59e3dcedd8c48715613d56eef702`; analyzer SHA-256 is `bd34a07c8429762a406e47ceafad0c7fd20eee3869b48fb57fc04b4e749a7600`; portable launcher SHA-256 is `d262762675de081eae56727fafa19285de7ce79966c06623f1a16871cd0fd687`. A non-counted two-epoch smoke passed with zero leakage and degree-2 decoder fidelity `0.999258`. Formal 15-cell execution is active in tmux `starry_zt_scale_bridge_20260829`, four CPU workers, repository-local caches and output root `runs/starry_zt_scale_aware_neural_bridge_20260829/`. The Matplotlib smoke created one automatic `/tmp/matplotlib-y9b2tml5` cache because of an inherited unwritable default; that exact directory was immediately deleted, and the formal launcher binds caches below the run root. No project artifact remains there.

## 86. Scale-aware neural bridge terminal: diagnosis validated, strict tail still fails (2026-08-29 02:20 CST)

The scale-aware campaign completed 15/15 cells successfully with finite outputs and exact zero query-target input difference. Degree-2 functional pooled R² is `0.942488`, physical RMSE `0.068765`, entity-bootstrap interval `[0.884230,0.974402]`, and minimum decoder-response fidelity `0.984515`. Structure re-q remains `0.980668`. Thus pooled expression accuracy and decoder fidelity remain far above their gates.

The one-factor target transform materially repairs the diagnosed failure: functional NRMSE improves in 59/80 entities; median entity R² rises from `0.846748` to `0.940261`; entities with individual R² at least 0.85 rise from 40 to 52; ten-times tail failures fall from 19 to 9; maximum ratio falls from `744.872` to `139.163`; and target-std versus functional-NRMSE Spearman weakens from `-0.614885` to `-0.193718`. Functional distance geometry now exceeds raw q at the median (`0.835419` versus `0.748142`) and in 11/15 seed pairs. Named coordinate stability is `0.819800` versus raw unaligned `-0.065064`, improving in 15/15 pairs.

The strict result remains FAIL because nine entities still exceed the frozen ten-times ratio. Seven improve, but two higher-scale entities worsen, so stronger uniform target compression is not an authorized automatic next step and may trade one tail for another. `scale_aware_tail_repair_supported=false` and `full_neural_to_canonical_bridge_supported=false` remain authoritative. Under the user's separate mandatory expression definition—compact, stage-wise suggestive, entity-held-out physical expression with pooled R² at least 0.85, without original-q or unique-law recovery—the endpoint remains passed; do not allow the stronger tail gate to erase it. Full results are in `runs/starry_zt_scale_aware_neural_bridge_20260829/SCALE_AWARE_BRIDGE_RESULTS.md` and `analysis/decision.json`.

`NEURAL_TRAINING_DYNAMICS_FOR_LATENT_Q_20260825.md` section 13 now derives the local weighting induced by `z=asinh(y/s)`: squared transformed error is approximately `(delta y)^2/(s^2+y^2)`. It connects this gradient reweighting to the observed low-scale tail repair and the two higher-scale regressions, and explicitly warns against tuning a smaller scale on the same 80 query entities.

## 87. Fresh ICLR audit: expression endpoint accepted, unified-method novelty remains weak (2026-08-29 02:25 CST)

A fresh read-only Luna reviewer estimated only `20%` probability (range `15%--30%`) of the current package entering ICLR's top half. Its strongest rejection is that the independently confirmed Starry equation is support structure re-q rather than a formula automatically transferred from learned raw q; direct raw-q reading fails, the neural bridge still has 9/80 strict tail failures, and prediction superiority over kNN is not significant. The correct defense is not to dispute these facts: keep the expression-existence endpoint separate, present gauge failure and canonicalization as the method insight, disclose tails/baselines, and avoid SOTA or true-law language.

The reviewer suggested a secondary scale-aware neural-bridge audit on the already consumed 30-material temporal cohort. This suggestion is explicitly rejected because the consumption receipt and both bridge plans forbid reopening those targets. The only valid stronger confirmation requires a genuinely new target-blind cohort with the complete decoder/calibration/projection package frozen first. Without such data, the correct paper action is to use the honest spine recorded in `ICLR_FRESH_REVIEW_AUDIT_20260829.md` and treat the neural bridge as supportive development evidence. The active top-50 goal is therefore not complete.

## 88. Independent ThermoML expression development passes; confirmation remains sealed (2026-08-29 03:10 CST)

The independent-domain follow-up uses the official NIST ThermoML 2020 archive and pure-substance liquid vapor-pressure curves. A target-blind audit initially selected 373 development and 116 temporal-confirmation compounds, but development materialization correctly failed before writing data because 101 selected development tables contained duplicate temperatures. No duplicate pressure was averaged or silently removed. A target-blind protocol correction now retains only tables with `rows == unique_temperatures`; the invalid first cohort seal is preserved at `runs/thermoml_vapor_pressure_cohorts_20260829_invalid_duplicate_temperature_protocol/`. The corrected cohort has 282 development compounds/142 DOI/9,794 rows and 84 temporal-confirmation compounds/45 DOI/2,372 rows, with zero InChIKey and DOI overlap. Confirmation JSON targets have not been opened.

The corrected plan SHA-256 is `8793f712b6a32aa514906ffb13ae7169d0de8556f9bda342b1202d94b0bb2deb`; cohort selection-manifest SHA-256 is `ed866c015e4017532a31251f62ff57ee26311f78232acbc729ec7ae91d8525c4`; development data SHA-256 is `9ebc8ea5a8b870cb98cc829c1700d4ebdad806c043014a0a5051ada8629411b6`. The materialized development data contain 2,554 support and 7,240 strict query rows, all positive, spanning `5.13e-15` to `8932.1` kPa. A pre-result amendment, SHA-256 `18e15a3bad4e5fc00464e1c4829062bbc5e7fed477d96781d1629b6884526faf`, records the user's durable expression criterion: stage-wise scientific interpretation plus strict unseen-entity pooled physical R² at least 0.85, finite positive predictions, and exact zero query-target leakage. Original-q recovery, a unique true law, prediction superiority, log-R², per-entity fraction, bootstrap, and PCHIP tail ratios are separate stronger evidence or diagnostics.

The frozen DOI-grouped five-fold structure loop selected
`ln(P/1 kPa)=q0+q1*(1/T-1/T_ref)+q2*ln(T/T_ref)`. The main term is Clausius--Clapeyron; `q0` is reference log pressure, `-R*q1` is an effective vaporization-enthalpy scale, and the logarithmic-temperature term is a stage-wise integrated heat-capacity/temperature-dependent-enthalpy correction. On all 7,240 development queries it reaches physical pooled R² `0.9998666486`, log pooled R² `0.9999766070`, median entity physical R² `0.9998779623`, and 282/282 entities at R² at least 0.85. Entity-bootstrap physical R² is `[0.9997200908,0.9999383972]`; DOI-cluster bootstrap is `[0.9996869835,0.9999598639]`; query-target perturbation changes support coefficients/predictions by exactly zero. The development expression endpoint passes. Authoritative decision/manifest SHA-256 are `928ee5bc4f21c156a00a49737dca0729b8e9a52f7fb23a165a870b7727adaa26` and `88a17aa99a58a9c265b6b6cace7fbd89e94f1544853e2f5b4669c07b36cd2485`.

Strong same-support baselines are now complete except the condition-only no-q MLP. Physical pooled R² values are selected expression `0.9998666486`, support Antoine `0.9998448263`, support PCHIP `0.9998136925`, support linear log-P `0.9996213216`, support nearest `0.9801594990`, and no-q global selected formula `-0.0985381654`. PCHIP has slightly lower median entity NRMSE (`0.0090324` versus expression `0.0110470`), while the expression passes entity R² 0.85 for 282/282 versus PCHIP 280/282 and has a much lower maximum entity NRMSE (`0.2711` versus `1.5794`). The expression beats PCHIP on 52.84% of entities. The legacy maximum expression/PCHIP NRMSE ratio is `141.49`, failing its old 10x diagnostic because several PCHIP denominators are nearly zero; the worst-ratio expression entity itself has NRMSE only `0.0153`. Do not mislabel this denominator pathology as an expression catastrophic tail, and do not infer broad predictive superiority. Baseline result/manifest SHA-256 are `d6ac603aaee7b66d6e63734586ce83e6b646537f13821b6c39ebaee994af0524` and `8bc59c2a666a5f3015a12c2948780bc37ccbfece6da3e3552c6f096e0e2218c4`.

Next boundary: do not open the 84 temporal-confirmation targets yet. First freeze the all-development reference temperature, selected expression, support fitting, complete baseline/no-q MLP behavior, and ThermoML neural-to-equation decoder/calibration/projection package. Only then create a single-use confirmation receipt and evaluate the unchanged package once. The current `.venv` was minimally populated with NumPy/Pandas/SciPy using repository-local `runs/_runtime_cache/uv`; the earlier full `uv sync --extra experiments` was interrupted during the large Torch wheel and installed nothing.

## 89. ThermoML neural bridge formal development chain active on CPU (2026-08-29 03:18 CST)

The operational supplement `THERMOML_NEURAL_BRIDGE_EXECUTION_PLAN_20260829.md` was frozen at SHA-256 `19265c4bc28ff7dba3f2a019af488b2db9c7b80fdbd3172607ed4c07f48611cb` before neural development results. It fixes log-pressure training/projection, 41 physical-range probes, physical projection-fidelity scoring, training-entity LOOCV raw-q ridge selection, exact support calibration, artifacts, and the rule that formal cells cannot mix CPU and CUDA. Formal runner SHA-256 is `e0374be1c4e271de8938f17e65ba05673cf34d018b2f99ea721fd6b71aa222bf`; four-worker launcher SHA-256 is `5860e20c30b25a1f88c233bf0e95202e33b64a42f83d3bccf690db22a0d77ec4`.

A non-counted fold0/seed0 two-epoch CPU smoke completed with exact zero query-target input difference, 1,450 query rows, structure re-q physical R² `0.9999720`, raw decoder `0.08051`, functional selected-expression projection `0.08040`, and minimum/median physical decoder projection fidelity `0.8091/0.9710`. The low two-epoch neural score is only a wiring check and must not be quoted as scientific evidence. Smoke manifest/summary SHA-256 are `6d3d041743c8d5068a3870da308ad375a65f2fa019af46f910579b3d3059b180` and `c4815e431ac5466c71bfbce46bdb892a3d4d6d4320d5aa3a327516c68aaf36de`.

The container still cannot communicate with the NVIDIA driver (`torch.cuda.is_available=false`, zero devices), so formal execution is uniformly CPU, not VRAM-capped. Tmux `thermoml_vp_bridge_20260829` runs 15 cells (5 DOI folds x 3 seeds), 1,000 epochs, four deterministic 1,200-step support-calibration starts, four workers x four Torch threads. At this timestamp fold0 seeds0--2 and fold1 seed0 have `manifest.running.json`; 0/15 are terminal and the launcher reports no failure. Authoritative live status is `runs/thermoml_vapor_pressure_neural_bridge_20260829/launcher_status.json`; logs/caches/checkpoints are repository-local. The 84-compound temporal confirmation remains sealed and unopened.

## 90. Matched ThermoML no-q baseline active; aggregate validators frozen (2026-08-29 03:27 CST)

`THERMOML_NO_Q_MLP_PLAN_20260829.md` is frozen at SHA-256 `9a5ba993425ffb9cabe2e4156cd12a3d4ee3ff37c0812f78e7b21e5a86fd0edf`. It uses the identical DOI folds/seeds, temperature-only explicit input, log-pressure target, `(256,128)` network, Adam `1e-3`, 1,000 epochs, batch 256, and per-minibatch label-balanced MSE. It sees neither entity ID/q nor heldout support/targets. Formal runner SHA-256 is `cebe5b206975e991922dc33619f4900fdce9096abebbb10449ec1dd92d9058ea`; launcher SHA-256 is `77911e7bff98843971fd16205de0c076bcd5698f5771c3eef84fb7ede1a60c67`.

A non-counted two-epoch fold0/seed0 no-q smoke passed with exact input-only query-target invariance, 1,450 queries, physical R² `-0.07104`, log R² `0.00496`, and 0.60 s training. These scores are wiring-only. Smoke manifest/summary SHA-256 are `7566b14206c14017a8c54aff57a1abb769b471af6831fd385dac6f33f491078d` and `a76d80c69112c97eef6d55000408c55867ce46c4b682da7a66b0b3bc2fb11dfc`. Tmux `thermoml_noq_20260829` now runs four CPU cells concurrently; 0/15 are terminal, 0 failed, 11 pending at this timestamp. Live status is `runs/thermoml_no_q_mlp_development_20260829/launcher_status.json`.

The independent neural and no-q aggregate analyzers are frozen and compile-clean at SHA-256 `f6a86fdaaef286e4f7d748915b9cf4d2fcfbc9647548ba01217411547e7c9719` and `e56b05066e1c8f783219a4d9649a800c001a24410679f7e2fc57cc8ef2b6886c`. The neural analyzer verifies every cell/artifact hash, uniform device and three-seed coverage, exact reproduction of the already sealed structure re-q predictions, physical/log metrics with entity and DOI bootstrap, decoder projection fidelity, raw-vs-functional cross-seed distance geometry, and leakage. It cannot run until all 15 cells are terminal. The no-q analyzer similarly requires all 15 matched cells and exact heldout-boundary evidence. The aborted `uv sync` byproducts `latent_variable_discovery.egg-info/` and untracked `uv.lock` were confirmed as newly generated environment artifacts and removed; no pre-existing worktree change was cleaned.

## 91. ThermoML physical q is support-stable, with explicit collinearity boundary (2026-08-29 03:34 CST)

The diagnostic plan `THERMOML_Q_STABILITY_PLAN_20260829.md` was frozen before offset results at SHA-256 `b99d58d317667be7a9bf27bb3ad5264ca380625066729f400bf649bcada2200a`. It keeps the selected `v_log` expression and DOI-fold reference temperatures fixed, then fits offsets 0--3 separately. Pooled physical R² remains `0.99986665/0.99985821/0.99979710/0.99985493`; pooled log R² remains `0.99997661/0.99992770/0.99992453/0.99991362`. No confirmation target was opened.

Across all six offset pairs, median/minimum three-coordinate physical-q distance Spearman is `0.89054/0.85462`. Named-coordinate median Spearman is `0.99905` for reference log pressure, `0.97619` for effective reference vaporization enthalpy `R(q2*T_ref-q1)`, and `0.79736` for effective heat-capacity correction `R*q2`. Offset-0 q distance has Spearman `0.56476` with normalized 41-point empirical log-pressure response-curve distance. Effective vaporization enthalpy median and 5--95% range are `56.23` and `23.12--128.74 kJ/mol`; effective heat-capacity correction is `-58.08` and `-224.55--12.53 J/(mol K)`.

The scaled design condition number is nontrivial: median `134.99`, 95th percentile `917.23`, maximum `6417.15`. Therefore the durable scientific claim is that reference pressure and effective enthalpy are highly stable response coordinates and the heat-capacity term is a weaker but reproducible curvature correction. Do not treat every entity's q2 as a precise calorimetric measurement or ignore local `1/T` versus `ln T` collinearity. Result/manifest SHA-256 are `4cc10db7d988f400f93574ff9a514dd05ed28f5c5c2b06b1dac08bd4caaaccb9` and `09b708dadded230ed9a6673a8f98ec7988ddf0bab7436962c01bcd2ebbdd4241`; analyzer SHA-256 is `2e84960b2e175bec31fb968cb8bacb3e3251ce35103954fab920f550c585ef11`.

## 92. Matched ThermoML no-q MLP terminal: support-derived state is essential (2026-08-29 03:43 CST)

The matched no-q chain reached 15/15 terminal success with 0 failures. Every cell used the identical DOI fold, 1,000 epochs, `(256,128)` widths, batch 256, Adam `1e-3`, label-balanced log-pressure MSE, and CPU device; every manifest states that heldout support values and heldout targets were unused. Three-seed pointwise median physical pooled R² is `-0.0992534`, log pooled R² `0.0150536`, median entity physical R² `-3.00575`, median/max entity NRMSE `2.00142/1145.62`, and only 1/282 entities reaches R² at least 0.85. Entity-bootstrap physical R² is `[-0.127666,-0.076710]`; DOI-cluster bootstrap is `[-0.142172,-0.069640]`; query-target input difference is exactly zero.

This closes a fairness concern: the selected support expression's `0.9998666` is not reproduced by a same-architecture, same-epoch temperature-only network. Entity-specific support contributes indispensable reference-pressure, effective-enthalpy, and curvature information. Do not infer that the expression is universally prediction-optimal; PCHIP and Antoine remain the relevant support-aware comparators. Authoritative no-q decision/manifest SHA-256 are `35a237cb60747207a1056449317ed4a34cb0a7da25a986a30086420b6ac89fe4` and `73a6a7bdbcff59802596c634f19b0807d5fc3f7a67a3a3767ac71a770e5c5d57`.

## 93. New pytest scratch migrated from `/tmp`; repo-local test temp enforced (2026-08-29 03:52 CST)

A fresh audit found no new project experiment, dataset, model, plot, log, or cache payload under `/tmp`. The only clearly attributable new material was 24 small pytest fixture files under `/tmp/pytest-of-wangyg` (96 KiB before migration), created by the MATR Batch2 analyzer tests after the earlier migration. With no pytest process active, the complete fixture root was moved recoverably to `runs/_tmp_archive_20260829/pytest-of-wangyg`; the source path is absent and the archived root contains all 24 files.

Future pytest-owned temporary paths are now fixed by `pyproject.toml` to `runs/_runtime_cache/pytest`. The one explicit `tempfile.TemporaryDirectory` test also uses that repository-local parent. The targeted protocol/metrics suite passed 15/15 under repository-local pytest, Python, Matplotlib, and XDG caches, and `/tmp/pytest-of-wangyg` did not reappear. The empty legacy Claude task directory, active Codex synthetic mount points, IPC sockets, general user caches, and unrelated projects were not moved. Durable rule remains unchanged: project outputs and fixtures go under `runs/`, external datasets under `data/external/`, and caches under `runs/_runtime_cache/`; never place project material in `/tmp`.

## 94. ThermoML single-use temporal confirmation passes the expression endpoint (2026-08-29 04:23 CST)

The 15-cell ThermoML neural development bridge and all three all-development seeds reached terminal success before confirmation was opened. Development decoder-functional `v_log` projection has physical pooled R² `0.9911733`, raw decoder `0.9905813`, direct raw-q ridge expression `0.2204505`, and support structure re-q `0.9998666`. Functional median entity R² is `0.993419` and 96.8% of entities reach 0.85, but the frozen full neural bridge fails: minimum decoder physical projection fidelity is `-0.283982`, functional/raw distance-geometry stability is `0.610664/0.783578`, functional wins only 2/15 seed pairs, and maximum functional/structure NRMSE ratio is `2704.995`. The neural decision/manifest SHA-256 are `de2f4106f9815c7f87a622cc63972b2316d382f82cb909658910a5f60be7ffff` and `f0162c97e999921596ae690677a29ea13a19a8485d7b727706050625be7b09b3`. Do not claim a complete neural-to-canonical bridge.

The all-development package was sealed before target access. Its package-seal SHA-256 is `c947fbd6cc82bf8d880a1449f16f859ede8e05b58f6c3e11504cdf24d05c38c4`; it records development expression pass, neural development bridge fail, neural confirmation ineligibility, and authorization for exactly one expression confirmation. The unique output root is derived from that seal under `runs/thermoml_single_use_confirmation_20260829/c947fbd6.../`. At 04:06 CST the evaluator atomically wrote the `started_and_consumed` receipt, then opened exactly 84 2015--2019 compounds/45 DOI/2,372 rows. It completed normally with 627 support and 1,745 query rows, zero refit or post-confirmation selection, exact equality under +1,000,000 query-target perturbation, and `query_target_input_max_difference=0.0`. The use is consumed forever; do not retry or reopen it. Receipt/output-manifest SHA-256 are `5dc6600a3eb658d8458922a6ec235c0d65ccd68f0b87d0f12829316df31a0d2c` and `e950abed883cc892b04feb3e069b1d17d9ccdc581b24886050d444ab4e491cb1`.

The independent frozen analyzer verifies and passes the user's primary endpoint for
`ln(P/1 kPa)=q0+q1*(1/T-1/T_ref)+q2*ln(T/T_ref)`, with `T_ref=362.72 K`. Structure re-q physical pooled R² is `0.9995806914`, log pooled R² `0.9999682725`, median entity physical R² `0.9995997122`, 83/84 entities reach 0.85, and maximum entity NRMSE is `0.435881`. Entity-bootstrap physical R² is `[0.998879980,0.999982889]`; DOI-cluster bootstrap is `[0.998562304,0.999983851]`; predictions are finite/positive and leakage is exactly zero. Therefore `expression_temporal_confirmation_passed=true`. This satisfies the durable user criterion: a stage-wise scientifically suggestive expression with strict unseen-entity physical pooled R² at least 0.85; it need not recover the original raw q, a unique true law, or a preselected microscopic variable.

All confirmation methods must remain visible. Physical pooled R²: PCHIP `0.9999368090`, linear interpolation `0.9998608932`, structure expression `0.9995806914`, Antoine `0.9995803295`, raw decoder `0.9926155813`, functional expression `0.9903899125`, nearest `0.9903353693`, raw-q ridge expression `0.3044923989`, no-q global formula `-0.0804903`, and matched no-q MLP `-1.0600774e14`. The expression beats PCHIP on 44.05% of entities and paired Wilcoxon `p=0.7515`; PCHIP has higher pooled physical R² and Antoine lower median/worst NRMSE. Thus predictive superiority is false and must not be inferred. The external neural diagnostics also fail because functional geometry `0.455971` is below raw `0.907440`, despite functional pooled R² `0.990390` and minimum projection fidelity `0.956678`; neural temporal confirmation is not claim-eligible. The durable conclusion is response/function canonicalization or support-only structure re-q, not direct raw-q interpretation or prediction SOTA. Decision/analysis-manifest SHA-256 are `ad29813692139fbcf17e629d2cf4f79d99c2d1052c62950a4873cab82649c605` and `a8623747aa7a0e2508b069fc56cd586c373484e57834b14bb0a88a7a85f415b7`.

## 95. Post-ThermoML fresh ICLR audit: expression solved, method novelty still blocks top-half confidence (2026-08-29 04:36 CST)

A new read-only Luna reviewer, instructed to ignore the previous audit and inspect the updated ThermoML raw decision, still assigns only `20%` probability (subjective interval `15%--30%`) that the current package reaches the top half of ICLR submissions. It estimates ThermoML increased that probability by only `3--5` points because the independent domain, 84 compounds, 45 DOI, temporal split, bootstrap, and zero-leakage receipt make expression existence highly credible, but do not close the learned-method contribution. The strongest rejection is that the confirmed endpoint is support structure re-q rather than learned raw-q-to-equation recovery; PCHIP/Antoine remain equally or more predictive; and the full neural bridge fails. This audit is archived in `ICLR_POST_THERMOML_AUDIT_20260829.md`.

The top-50 goal remains active even though the user's mandatory interpretable-expression goal is now genuinely achieved. Do not spend the next cycle merely accumulating another similar high-R² curve fit. The highest-value route is to formalize decoder-response canonicalization as a gauge-invariant method with an explicit theorem/stability bound and a controlled gauge intervention benchmark showing that prediction-equivalent raw q coordinates change while fixed-basis functional coordinates do not. A later genuinely new target-blind entity/domain comparison should include test-time auto-decoder, support encoder/CNP, kNN, FPCA or function-space baselines, tails, stability, and compute. If a new cohort does not pass a full neural-to-functional bridge, limit the paper to gauge diagnosis, decoder-function projection evidence, and support-only structure re-q.

## 96. Gauge-invariant canonical-response benchmark frozen and active (2026-08-29 04:46 CST)

The next method-focused experiment was frozen before formal results in `GAUGE_INVARIANT_CANONICAL_RESPONSE_BENCHMARK_PLAN_20260829.md`, SHA-256 `ba2a587bd6f7a2945b118c2316ae8f52e0dce9663abfb2fe03f81a084720ada6`. It uses three fixed-basis synthetic response families, 96 training and 48 identical unseen test entities per family, five optimization seeds, support positions 0/4/.../40, 30 strict query positions, a three-dimensional auto-decoder, 1,500 epochs, four-start 1,200-step support-only calibration, and 25 exact affine latent-gauge interventions per cell. The primary theorem test requires prediction and decoder-functional coefficients to change by at most `1e-5` under the exact gauge, all functional and structure family query R² values at least 0.85, median functional-vs-generating coordinate Spearman at least 0.90, and exact query-target invariance. It explicitly does not claim automatic basis discovery, unique/causal latent recovery, nonlinear-gauge universality, or prediction superiority.

`LATENT_Q_INTRO_MOTIVATION_THEORY_RELATED_WORK.md` now states and proves: (1) fixed-basis decoder-response coefficients are constant on the latent-gauge equivalence class; (2) an exact first-layer affine counterfactual `Wq'=Wq A^-1`, `a'=a-Wq A^-1 b`; (3) decoder projection coefficient error is bounded by the weighted response error divided by the smallest weighted-basis singular value; and (4) function-induced coefficient geometry is invariant to invertible basis-coordinate changes. This formalizes why original raw q recovery is unnecessary while still exposing basis misspecification and collinearity.

Runner/analyzer/launcher/test SHA-256 are `163e5d833152850e775c70ca2ffcdfc0aa9122be82884bfcae0db3aeb5b12b1f`, `bbf800f9970ef36e6c77330615402a6545dd7f75109b63fafd27051ff3484d15`, `7fb46740acda906813441bc6e3b5938bacc08975b98da3b7ded0f1a3963b7849`, and `c309964f6b68f3ec1fc9516dc3152ab6422542b83f7f4b116c745f446b7e787a`. Six focused tests and compile/diff checks pass. The non-counted 2-epoch/2-step smoke took 6.77 seconds, had maximum gauge prediction change below `5e-16`, functional coefficient change below `1e-15`, material raw-q/readout change, and zero query-target perturbation difference. No ThermoML confirmation target was read.

Formal execution is active in tmux `gauge_canonical_20260829`, output `runs/gauge_invariant_canonical_response_benchmark_20260829/`, with eight concurrent CPU cells on a 128-core host (load about 11 at launch, user CPU about 0.3 core before launch, 638 GiB available memory). The container still cannot use CUDA, so all formal cells are uniformly CPU. The launcher runs all 15 cells and invokes the independent analyzer only after all succeed. At launch the first eight logs exist and are empty because each cell writes only at terminal completion. Do not promote smoke or partial cells to scientific claims; inspect the 15 manifests and final `analysis/decision.json` first.

## 97. Exact gauge algebra passes; strict audit narrows the claim and freezes calibration extension (2026-08-29 04:57 CST)

The formal exact-affine benchmark completed 15/15 CPU cells with 15/15 formal eligibility, zero failures, six focused tests passing, and terminal independent analysis. Five-seed median decoder-functional physical pooled R² is `0.9999392726` for polynomial, `0.9998066844` for relaxation, and `0.9998379485` for the thermodynamic chart; support structure re-q is `0.9999701855/0.9999580994/0.9999759115`. Median functional-vs-generating coordinate Spearman is `1.0/0.9993487/0.9598350`. Across 375 exact affine interventions, maximum prediction change is `4.4964e-15` and maximum functional coefficient change `2.0428e-14`, while raw-q coordinates change by up to `4.99582` and an intentionally unsynchronized frozen raw readout changes predictions by up to `33.4874`. Query-target input difference is exactly zero. Every frozen primary gate passes; predictive superiority and unique/causal recovery remain false. Decision/analysis-manifest SHA-256 are `2d31a2f3b0e3d95018a2bd1c99f9bbf575bc59cc351ebc984c97268de25bca59` and `2d77ffae6d2dfd9960310c48f2e90cfa0ce5e42706d420366556d14bf5d43a3e`.

A fresh Sol xhigh mathematical audit judged the core algebra correct but the empirical claim too broad. The original benchmark calibrates q first, then hand-constructs `q'=Aq+b`; it therefore proves an algebra/implementation gate, not that independently rerunning Adam on transformed coordinates returns a paired response. The audit also identified missing theoretical conditions and analyzer mismatches: global entity-independent `T`, response equivalence rather than directly observed full functions, `W^(1/2)Phi` full rank, basis-relative rather than absolute canonicality, calibration-selector caveat, true-coefficient residual bound, support rank/query residual terms, and function-induced rather than raw coefficient Euclidean geometry. `LATENT_Q_INTRO_MOTIVATION_THEORY_RELATED_WORK.md` now includes these corrections, the exact affine architecture assumptions, and ideal-gas/finite-temperature ThermoML assumptions. Do not cite the exact-gauge experiment as independent calibration equivariance or as sufficient ICLR novelty.

Before inspecting any formal scientific metric, the separate `GAUGE_EQUIVARIANT_CALIBRATION_AMENDMENT_20260829.md` was frozen at SHA-256 `b274f1abaee71990c5a78152d92070262caa6c37572367c60167c7bbe8fbc91f`. It requires independent original/transformed calibration from each chart's transformed training-embedding centroid, comparing mapped-start Adam with a response-metric Gauss--Newton method. The latter uses 30 full-rank support-Jacobian steps and coordinate-free support-loss line search; under `J'=J A^-1` its step transforms as `delta'=A delta`. The extension also requires basis-H covariance, probe-response geometry, per-entity projection/support bounds, narrow-support diagnostics, synchronized raw readout, and exact query-target invariance. Primary gates include maximum independently recalibrated GN response/functional difference `<=1e-6`, all family functional R² `>=0.85`, basis errors `<=1e-8`, and zero deterministic-bound violations. This is a new output root and cannot overwrite or reinterpret the algebra result.

## 98. Expression endpoint reconfirmed; target-blind next-domain audit identifies crystal heat capacity (2026-08-29 05:13 CST)

The user reconfirmed the durable expression endpoint without adding a stricter latent-recovery condition: a comparatively interpretable expression is successful when its strict unseen-entity/support-query prediction in physical units reaches pooled `R² >= 0.85` and the relation offers a plausible stage-wise scientific clue. It need not equal the initial raw `q`, recover the originally imagined physical quantity, or be the unique true law. Decoder-functional coordinates and structure-recalibrated `q` are both eligible. This is already stated exactly in `COMPLETE_RESEARCH_REPORT_20260809.md` sections 22.4 and 24, so the report was not redundantly rewritten. Predictive superiority over PCHIP, kNN, FPCA, no-q MLP, or other support-aware baselines remains an independent stronger claim and cannot be inferred from this expression endpoint.

A response-value-blind ThermoML metadata audit discarded every `PropertyValue`/`nPropValue` before candidate selection and excluded the already used vapor-pressure property. Under the conservative single-component, single-property, temperature-only, positive-temperature, exact `Crystal` phase, fixed `101.325 kPa`, DOI/InChIKey-disjoint temporal protocol, crystal molar heat capacity is the strongest new domain: 247 development compounds/159 DOI/23,742 rows through 2014 and 86 confirmation compounds/62 DOI/11,447 rows in 2015--2019, with at least 20 unique temperatures per curve. Crystal molar entropy is a secondary eligible domain at 114/46 development/confirmation compounds; strict-phase molar enthalpy has only 86 development compounds and does not meet the 100-compound floor. The first-choice frozen physical family should be Shomate-like `Cp(T)=A+B t+C t^2+D t^3+E/t^2`, `t=T/1000`, compared in physical units against support-nearest, support-linear, PCHIP, support-ridge in the identical fixed basis, an all-development no-q global formula, and a temperature-only no-q MLP. No response value has been inspected, so this remains target-blind candidate selection rather than a scientific result.

## 99. Independent calibration formal failure exposes finite-precision condition; stable amendment and GIRD frozen (2026-08-29 05:45 CST)

The first independent recalibration extension is terminal at 15/15 formal CPU cells and its independent analyzer is terminal. Formal runner/analyzer SHA-256 are `13f9d21d1525582a2bb874add150bf5679f41642486b62dbbf8c63a2e3286024` and `3e48417ba04ed991a2e091c2c9ee7bdc80e5de06f4fa7a018e5b5bf215db3492`; eight focused tests passed before launch. All three original-chart Gauss--Newton functional-expression pooled physical R2 values pass (`0.99993927/0.99980668/0.99995121` for polynomial/relaxation/thermodynamic chart), all support Jacobians are rank three, basis-H coordinate/fitted-response/pair-response maximum errors are `2.88e-14/2.12e-13/1.88e-13`, both deterministic bounds have zero violations, and every query-target perturbation field is exactly zero.

The overall frozen benchmark nevertheless fails and must remain reported as a failure. Maximum independently recalibrated Gauss--Newton query-response and functional-coordinate differences are `0.0448353507` and `0.1504709407`, above `1e-6`. Failure is concentrated in thermodynamic-chart seeds 3/4, whose Jacobian condition reaches `61438.78`. The original implementation forms `J^T J`, squaring conditioning; the worst paired path agrees early, begins floating divergence near iterations 19--20 and selects different line-search steps at iterations 26--29. Polynomial/relaxation remain near `1e-8` or lower. Thus exact-arithmetic equivariance is not yet a robust finite-precision implementation. Authoritative decision/analysis-manifest SHA-256 are `2214a5ff161573d2d9ba767e1d8dd60134ab536500979dd068c59dd4038d49f0` and `17ce41703ea7d305d598dc14d877d28c29580f057c1f2bd23b620ab5152e4fc4`. Do not overwrite or relabel this root.

The post-failure numerical amendment `GAUGE_EQUIVARIANT_CALIBRATION_NUMERICAL_AMENDMENT_20260829.md`, SHA-256 `d85db0c6d9a5b332aa3499ebd3f105a2e89a4674b30fe90db0657bd26006613`, was frozen before any corrected formal result. It uses direct float64 SVD/QR least squares, response-loss acceptance margin `1e-12*max(1,loss)`, exactly 15 iterations and a separate root while preserving every original scientific threshold. It adds exact step-scale pairing and at least 100x improvement gates. The new stable runner/analyzer/launcher/tests are respectively SHA-256 `299d47d830ae7a8e4402db81bbf94a60ac64b23450ed04c55e2dd7b5294316f7`, `6307f8e4398e9e91bf707e03d2a1b2d4af12e095f6737fcf3c11d83b76cdc469`, `4661006234cca4f39a2b939d2ffef4d8fdb13a84a146046db7edf4bb52ca9b8e`, and `8a8d653b3bdb91bed32d2696f14cc8a15e71e5080554c6e8c3b9f2b6079f7ccb`. Formal eligibility is bound to the exact stable formal root; the portable launcher derives its root from its own location and accepts `PYTHON_BIN`. Four focused tests pass, compilation/shell syntax/`git diff --check` pass. In two non-counted all-48 worst-cell diagnostics (current code, thermodynamic seed3), 15-step maximum response/functional differences were `3.6567271433e-10/6.2327461664e-09`, with exact line-search step scales across all gauges, full GN rank, and zero query-target perturbation. No corrected formal run has started.

The next paper method is frozen as Gauge-Invariant Response Dictionary Discovery in `GIRD_METHOD_EXPERIMENT_PLAN_20260829.md`, SHA-256 `c31f1fca60219f2cf2b258bac09e1a590dd7962c5e55d19f659730b743602072`. It selects at most five symbolic atoms by deterministic heterogeneous-design multi-response OMP from independently calibrated decoder probe responses, then performs basis-covariant decoder/support coefficient fusion. The decisive method gate is not another high pooled R2: in an exact four-support stress, finite `lambda>0` must improve median entity NRMSE by at least 5% over `lambda=0` and beat an identical direct-target dictionary. Synthetic controlled, vapor-pressure development-only and prospective crystal-Cp experiments are separated; the consumed vapor-pressure confirmation is never reopened. The matched ordinary symbolic-regression control is now frozen explicitly as per-entity support-only OMP over the identical atom library/conditioning/budget, distinct from the shared direct-target dictionary.

The target-blind crystal-Cp protocol is now version `thermoml_crystal_cp_target_blind_v2_gird`, SHA-256 `2ae03f71e6ffe9cfee3df0a61c8c7e49e9777268d0d9ccb6f1da8538e2203618`. It includes the exact 12-atom physical library, K/lambda grids, four-support split, Shomate/PCHIP/direct-target/SR/FPCA/CNP/auto-decoder/no-q baselines, three separate endpoints and a single-use temporal seal. No selected Cp response value has been read or materialized. `COMPLETE_RESEARCH_REPORT_20260809.md` section 25.3--25.4 and the theory document now preserve the formal failure, condition-number diagnosis, numerical amendment and GIRD boundary.

## 100. `/tmp` storage re-audited; persistent materials remain under the public project tree (2026-08-29 06:07 CST)

At the user's direction, `/tmp` was checked again before moving anything. No top-level path owned by this project and named for `lvs`, `matr`, `starry`, `thermoml`, `gird`, `gauge`, `latent`, or `pytest-of-wangyg` is present, and a content scan found no file referencing this repository, the gauge-calibration experiments, GIRD, or the ThermoML crystal-Cp plan. Therefore there was no new project payload to move and no unrelated `/tmp` entry was touched. The previously migrated recoverable material remains under the public repository at `runs/_tmp_archive_20260828/` (41 MiB) and `runs/_tmp_archive_20260829/` (133 KiB); runtime caches remain at `runs/_runtime_cache/`.

Durable rule: all persistent project scripts, experiment outputs, downloads, fixtures, logs, figures, and audit artifacts must be written below `/public/home/wangyg/latent_variable_search`; use `runs/`, `data/external/`, and `runs/_runtime_cache/` according to role. `/tmp` is not a project storage location. Current agents were explicitly notified of this rule. System IPC, sandbox mount points, unrelated projects, and ambiguous legacy files in `/tmp` must not be moved into this repository.

## 101. Stable independent-calibration formal run launched; GIRD decision statistics tightened pre-result (2026-08-29 06:20 CST)

The stable SVD/QR extension passed a second main-agent audit before launch. The launcher now checks that the explicit interpreter imports Torch/NumPy/Pandas before creating the formal root, waits for and records every owned child failure, and never runs the analyzer after a cell failure. The analyzer enforces the frozen 30 query positions, exact per-entity Adam/GN iteration sets, full method/gauge/basis/bound/geometry coverage, and recomputes all raw original-versus-query-perturbed path, calibration and prediction differences rather than trusting the runner summary. The raw recomputed maxima are part of the final leakage gate and a separate analysis artifact. Four focused tests, bytecode compilation, shell syntax, explicit environment import, and `git diff --check` passed. Final pre-launch hashes are runner `5257d739592caf96249eb6dc5e8bca81734bf744a20e6fa8366fbaa806150fbd`, analyzer `809b7514719ac7d31ce324c4536d7c5c1418c39fbdcba47e6b33d8dae09ca233`, launcher `ca693834509fef5772d13c8a49380bc4000c462c93711d5e6a3bec34e93cc663`, and test `fb4307a87d506e267f39835813f8a4ec793076b8a1b7affa9d7bd5fdacfb5a0a`.

The exact formal root was absent immediately before launch. At 06:20 CST tmux `gauge_stable_20260829` started the 15 formal CPU cells with `MAX_PARALLEL=8` and the explicit interpreter `/public/home/wangyg/latent_variable_search/.venv-lvs-gpu/bin/python`. Host load was about 13--15 on 128 cores with about 618 GiB memory available. The first eight launcher logs were created and the tmux session was live. Do not promote partial cells; wait for all 15 manifests, `analysis/decision.json`, its manifest, and `launcher.done`, then reconcile every gate with raw artifacts. The old failed normal-equation root remains authoritative negative evidence and must not be overwritten.

Main-agent review found that the first GIRD implementation did not yet meet the frozen plan: its analyzer reread runner-written OMP winners instead of independently recomputing from raw response/design inputs, used pooled rather than entity-normalized NRMSE for the decisive comparison, omitted several controlled gates, and misinterpreted the four-fold lambda rule. No formal GIRD root was created. Before any formal GIRD result, `GIRD_CONTROLLED_DECISION_STATISTIC_AMENDMENT_20260829.md` was frozen at SHA-256 `e110948ab67dec54b4b9d28c96c2276eb29912f6e988bd160bc7869236a04df0`. It fixes five-seed pointwise aggregation, entity-specific NRMSE, fold-specific lambda selection, a primary GN-only 4/5 finite-positive-lambda rule, deployment lambda selection, outer 48-entity median-NRMSE comparisons, exact remaining controlled gates, direct float64 least-squares handling of rank-deficient lambda-zero fits, and raw-input independent recomputation. The GIRD runner/analyzer are being revised against this amendment and remain ineligible to launch until the stable extension passes and their final hashes/tests are independently reviewed.

## 102. Stable independent calibration passes; expression criterion reaffirmed (2026-08-29 06:31 CST)

The stable SVD/QR formal extension is terminal at 15/15 cells and `launcher.done` exists. Independent manifest verification found 13/13 analysis artifacts and every cell-declared file hash exact, with no mismatches. All ten frozen primary gates pass. Five-seed pointwise-median functional-expression physical pooled R2 is `0.9999392728/0.9998066804/0.9999526932` for polynomial/relaxation/thermodynamic chart. The maximum independently recalibrated stable-GN query-response and functional-coordinate differences are `3.6567271433e-10` and `6.2327462011e-09`; every paired line-search step scale is exactly identical. The old response difference was `0.0448353507`, giving improvement factor `122610599.4`. Basis coordinate/fitted-response/pair-response errors are at most `2.92e-14/2.10e-13/1.87e-13`, deterministic bounds have zero violations, and all raw query-target perturbation maxima are exactly zero. Predictive superiority and unique/causal latent recovery remain explicitly false.

Two analyzer-only infrastructure failures occurred after all scientific cells had completed. The first referenced an undefined perturbation-artifact constant; the second applied GN-only finiteness requirements to intentionally NaN Adam diagnostic columns. Both logs are preserved under `launcher_logs/`; neither created an analysis directory nor reran a scientific cell. The corrected analyzer independently recomputes the raw perturbation audit and uses method-specific finiteness checks. Final SHA-256 values are runner `5257d739592caf96249eb6dc5e8bca81734bf744a20e6fa8366fbaa806150fbd`, analyzer `f0badabe855f01c3335955a013f051b430fa5eed98f1a9b8e4ca06bc2a74d557`, launcher `ca693834509fef5772d13c8a49380bc4000c462c93711d5e6a3bec34e93cc663`, test `e3afda79fdaccad9fd11635d5e656199cb2a2ad82ad28b362f41688db91c5bef`, decision `f3b1a222cf9e56b4209f0e6183dbd80792ae96738a905c06759fa4f655a1a1a0`, and analysis manifest `980bd9e7540cf733fbdeb5aba33ea7339b46c3968d53918836e6ab8eaf9fcc30`.

The user again reaffirmed the durable scientific endpoint: a compact expression counts as interpretable when strict unseen-entity/support-query physical pooled `R2 >= 0.85` and the relation offers stage-wise scientific inspiration. It need not recover the initial raw q, a preselected physical quantity, or a unique true law. Decoder-functional and structure-recalibrated q are eligible. This criterion was already frozen before the relevant real-data outcomes, so no frozen protocol was changed. Predictive superiority remains a separate stronger endpoint and must not be silently required or inferred.

## 103. GIRD preformal audit completed; one non-counted smoke exposes a hard lambda-zero risk (2026-08-29 06:52 CST)

No formal GIRD root existed during the audit. The runner now binds the exact passing stable-extension runner, decision and analysis-manifest hashes (`5257d739...50fbd`, `f3b1a222...a1a0`, `980bd9e7...cc30`). The inner lambda-selection paths no longer reuse one Adam prior for every method: each inner validation entity is recalibrated from support only with the matching 15-step stable GN or 300-step Adam method, while the true-basis reference uses the primary GN prior. The raw bundle records separate method/fold/entity probe priors with exact finite masks, and the independent analyzer verifies the 15/300-step coverage, GN ranks, paths, raw OMP inputs, lambdas, coefficient fits and predictions. The lambda helper now records the score of the 1%-tie-selected lambda rather than incorrectly attaching the absolute best score.

The analyzer now follows the controlled decision amendment exactly: the primary four-support comparison requires beating the direct-target dictionary; ordinary per-entity symbolic regression and FPCA remain reported comparators rather than silently added hard gates. Adam remains a diagnostic comparator; the primary OMP certificate gate is scoped to the GN response path, while Adam certificate status is separately written. The function-subspace recovery calculation now uses principal angles correctly when a selected dictionary is a higher-dimensional superset of the three-dimensional generating subspace, while a lower-rank candidate still fails. Recovery is independently recomputed from raw atom designs rather than trusted from runner flags. Twelve focused tests, compilation, launcher import/help and `git diff --check` pass.

One repository-local, non-counted polynomial seed-0 smoke completed both exact support regimes. It has zero query-target effect, correct separate inner-prior coverage, passing GN OMP certificates and generating-subspace containment. It selected five atoms containing `{1,x,x^2}`. Importantly, both standard-11 and four-support selected `lambda=0` for this one smoke cell. This is not a formal aggregate and must not be promoted, but it is a transparent warning that the frozen four-support 5% fusion gate may fail for an easy polynomial family whose support already identifies the true quadratic response. No atom, support position, lambda, threshold or scientific gate was changed after seeing this. The formal experiment must run unchanged and any failure must remain visible.

Final preformal SHA-256 values are runner `c8744aa8025dc94d83c43f3a3c2a19d43ecceb7a57505c2f5436376b91f5d3e7`, analyzer `7310d5f72525fa9a907d2781a52f74ffe8ae54191844d64bc177d3d384b37f62`, launcher `91f5df4f7e45f6f3ccf757d18b98e882b106c31aa53e8ed0127e342db74f6ba7`, runner test `34556cb60e4fe7448245af03a9e4ff82b9512b0aee6c17aed5e2507a7cb42168`, and analyzer test `d091b5391ec351513b50624e4752b9d9fa795b70fb28f09d3850e8945c939034`. The portable launcher runs 15 family/seed cells with configurable `PYTHON_BIN`/`MAX_PARALLEL`, collects all failures, invokes the analyzer only after all cells succeed and writes `launcher.done` only after analysis succeeds. The formal root remains absent immediately before launch.

## 104. Formal controlled GIRD launched unchanged (2026-08-29 06:42 CST)

Immediately before launch, the exact formal root was absent, host load was `15.83/14.57/14.92` on 128 cores, user processes consumed about 0.4 CPU core, and about 626 GiB memory was available. NVIDIA access remains unavailable inside this container (`nvidia-smi` cannot communicate with the driver), so this controlled experiment is CPU-only. Tmux `gird_controlled_20260829` launched the exact frozen 15 family/seed cells with `MAX_PARALLEL=8` and `/public/home/wangyg/latent_variable_search/.venv-lvs-gpu/bin/python`. The first eight runner processes were confirmed active. No partial result may be promoted; wait for all 15 parent manifests/results, independent `analysis/decision.json`, its manifest and `launcher.done`. If the frozen polynomial four-support lambda gate fails as warned by the non-counted smoke, preserve it rather than changing the formal protocol.

## 105. Controlled GIRD terminal: conditional fusion gain, universal gate failure (2026-08-29 07:10 CST)

The first launch failed after writing only all 15 `standard_11` results because a relative extension root was passed into an absolute `Path.relative_to` call.  The complete failed root is preserved as `runs/gird_controlled_discovery_20260829_infrastructure_failure_relative_extension_root_20260829_0644/`.  The runner was changed only to resolve the extension root at the system boundary and a regression test was added.  A clean absent formal root then completed all 15 family/seed parents and all 30 `standard_11`/`four_support` regimes.  No scientific cell was rerun after those 30 regimes became terminal.

The launcher analyzer exposed several analyzer-only defects: an undefined validation variable, a missing independently reconstructed validation count, an unbound fit-diagnostic frame, lambda-zero grid rows conflated with appended selected rows, a bad Pandas row/column index, and a relative run-root manifest path.  The original failure log and two pre-final analysis directories are preserved.  The final analyzer additionally recomputes and verifies dictionary validation rather than merely loading it.  Fourteen focused tests pass, compilation and `git diff --check` pass, and independent manifest reconciliation found zero cell or analysis hash mismatches.  Final runner/analyzer/decision/analysis-manifest SHA-256 values are `e6559560679583d7725c69cb35a3a2d6619f26f3ef0060fb98ea77a3017b51df`, `66b57f74c9200a647d0a65d6fc988ad4ad8ed3e262816c52fc40751c4fafb1c0`, `cf473722eae2cbeeda28b30e1584e120c0f06c45e1269761b2774050e18d77b1`, and `f2ff6b10d48b9bbc347de471a2a15b9c45d0c0818e65499aaff9b4b9fcee2198`.

The frozen universal GIRD gate fails.  Four-support polynomial selects lambda zero and stays at median entity NRMSE `0.010421`; thermodynamic chart also selects lambda zero and stays at `0.019240`.  Relaxation is the constructive positive case: selected fusion lowers median entity NRMSE from `0.052771` to `0.034355`, a `34.90%` improvement.  All three selected GIRD methods beat the direct-target dictionary but lose to FPCA.  All six GN functional/fused pooled physical R2 gates exceed `0.85`; functional-to-generating-coordinate median Spearman is `1.0/0.999240/0.990990` by family.  GN dictionary selection is identical under every independently recalibrated affine chart in 30/30 regimes, while Adam is stable in only 6/30.  Real remaining failures are: basis covariance whenever rank-deficient four-support fitting selects lambda zero, two of 30 GN regimes with an uncertified OMP selected margin, and standard-support GIRD lying about 5.6--17.4% above the best frozen interpretable control.

Durable interpretation: do not claim universal fusion gain or prediction superiority.  The next justified method hypothesis is conditional, rank/conditioning-aware fusion: use a gauge-invariant decoder prior when support fitting is underdetermined or ill-conditioned, and safely back off when support alone identifies the curve.  Because this hypothesis was induced by the terminal controlled outcome, it requires a separately frozen experiment on unseen synthetic families or a new real cohort.  The user's expression endpoint remains independent: strict unseen-entity/support-query physical pooled `R2 >= 0.85` plus a compact stage-wise suggestive expression is sufficient; original raw-q recovery and prediction SOTA are not required.

## 106. Crystal-Cp execution is unblocked at the protocol level, but response access remains sealed (2026-08-29 07:21 CST)

A fresh target-blind Luna red-team audit read no crystal-Cp response and found that the conceptual temporal/identity firewall was sound, but the execution chain was not yet launchable: the old ThermoML auditor only covers vapor pressure/viscosity, the expected crystal counts had not been reproduced by a formal selector, and Cp-specific materialization, baselines, neural/GIRD analysis, package seal and single-use confirmation code were absent.  It also identified underspecified CNP episodes, FPCA grid, PCHIP extrapolation, no-q selection, calibration starts, exact affine gauges, and the misleading `ordinary_symbolic_regression` name for a support-only OMP control.  Existing vapor-pressure scripts must not be relabeled or reused as Cp evidence.

Before any selected Cp response was opened, `THERMOML_CRYSTAL_CP_RANK_AWARE_GIRD_AMENDMENT_20260829.md` was frozen at SHA-256 `fffb406998900ff38131ee58bd9d98364ea05c3a334bf4b639c0456696c77639`.  It replaces only the universal four-support fusion gate.  A held-out entity is classified from support temperatures/model derivatives alone as support-identified, prior-eligible, or jointly unidentified.  The conditional rule uses exactly `lambda=0` when the symbolic support matrix is full-rank/well-conditioned, a development-selected finite lambda only when the symbolic support system is rank-deficient/ill-conditioned but the decoder calibration Jacobian remains identifiable, and retains jointly unidentified entities in all overall metrics.  A conditional-gain claim needs at least three contributing DOI folds, 20 prior-eligible entities from five DOI groups, at least 5% paired median-NRMSE improvement over visible lambda zero, and a win over the matched direct-target dictionary.  Insufficient stratum size is `NOT TESTED`, never a pass.

The companion `THERMOML_CRYSTAL_CP_EXECUTION_CONTRACT_20260829.md`, SHA-256 `ec37eff5ab2c5847735e4b3d8db4098fd4db2bcbf67792e4b54a4fb8ba43ea15`, freezes the response firewall, exact support indices, endpoint-tangent PCHIP extrapolation, kNN ties, normalized-grid FPCA, no-q selection, full-curve/sparse-support auto-decoder pathway audit, four deterministic Hadamard calibration starts, stable 15-step SVD/QR GIRD calibration, CNP episode/update accounting, eight exact affine-gauge families, and the `support_OMP` name.  The hash supersedes the immediately prior unexecuted contract draft solely because its prefix count was corrected from ceiling to the parent plan's exact `max(5,floor(n/4))`.  The parent target-blind plan remains unchanged at SHA-256 `2ae03f71e6ffe9cfee3df0a61c8c7e49e9777268d0d9ccb6f1da8538e2203618`; all three documents must be bound by future manifests.

At the same transition, the legacy 2,910-job campaign remained `completed_all` with zero failures and no related tmux/process was active.  A host-level read-only refresh showed GPUs 2, 3, and 4 at zero MiB and GPU 5 at four MiB, while 0/1/6/7 were occupied; no card was claimed because metadata selection is CPU-only.  A Cp selector and a CPU expression/local-baseline runner are being implemented in separate new files with synthetic fixtures.  No formal Cp output root exists, no development response has been materialized, and confirmation remains unopened.

The theory document now contains the exact response-metric shrinkage decomposition behind this rule.  After whitening coefficients by the probe Gram and writing a support singular direction as `z=s_i u*+epsilon`, `u_f=u*+eta`, fused risk is `(s_i^2 sigma^2 + lambda^2 tau_i^2)/(s_i^2+lambda)^2`, with isotropic oracle `lambda*=sigma^2/tau^2`.  Small support singular values amplify the support-only risk; large singular values justify backing off; a zero singular value means the decoder prior supplies an otherwise unidentified function direction.  This is a motivation and applicability condition, not a guarantee of real-data gain, so the frozen nested DOI validation and comparator gates remain necessary.

The formal target-blind selector then completed against all 11,923 source JSON files.  It lexically replaces `PropertyValue`/`nPropValue` payloads with null before JSON decoding; no response scalar enters the selection object graph.  A first implementation-level count discrepancy was diagnosed before any output existed: the frozen counts describe the deterministically selected one-table-per-InChIKey cohorts, whereas the initial check was mistakenly applied to the pre-dedup table pools.  Moving only the check to the already frozen post-tie-break layer yielded the exact expected selected counts without changing any cohort rule: development `247 compounds / 159 DOI / 23,742 rows`, confirmation `86 / 62 / 11,447`.  The visible eligible pools are `355 tables / 247 compounds / 162 DOI / 29,019 rows` and `147 / 86 / 65 / 15,349`; these explain the earlier diagnostic and are preserved.

Independent hash and identity reconciliation passed for all six selector artifacts, five DOI folds, and both selections.  Development/confirmation InChIKey and DOI overlaps are exactly zero.  Formal selector, selection-manifest, development-selection and confirmation-selection SHA-256 values are `d0668d13bcb3a56df4604151f96170c5a03a0703eed491504782345750a93727`, `8760bb5e2b09e5636178b97e3a7a7f9699c8e896a44b0cfa88a57116caeced33`, `a48e66da7b13e070f3f7c0b8622364ce448a0c350d54bf5e570b7ba84840e9c0`, and `f937b660fb9294821fc08f0b7a84ebc8d43f01c1e5c93bcb555d0cf5cd1d0ddd`.  Fold entity counts are `44/52/64/47/40`, with source rows balanced at `4,738--4,765`.  Confirmation response remains unopened; only development materialization is now authorized.

## 107. Crystal-Cp development opened; smooth Shomate fails pooled but reveals a transition-aware structure (2026-08-29 08:00 CST)

The development-only materializer passed its synthetic firewall tests and then opened only the 159 development source files selected by the sealed cohort.  Development/confirmation source overlap is exactly zero; no confirmation source was opened.  It materialized `247 entities / 159 DOI / 23,742 rows` with exact sealed temperature sequences and support indices.  Spread/prefix/four support counts are `6,038/5,845/988`, leaving `17,704/17,897/22,754` queries.  All heat capacities are finite and positive, spanning `4.8342e-6` to `57,525 J/K/mol`; no outcome-based row was removed.  Materializer/data-manifest/data SHA-256 values are `0e05a363abadec461e1326c0d5b7a1ee1d988d286752737ff29c2df85e894896`, `d88172109a7a244195f13dfa01516cfe6cb16cd038ca797dfd61d2d50694208a`, and `f73d3c676932304c8e5c21e79e7bc9c678e20c84db8d60b59a8e60feee400e4e`.

The first formal v2 CPU baseline attempt was deliberately interrupted before any file was written because its leakage audit repeatedly searched an ever-growing prediction table, producing quadratic runtime.  Its empty root is preserved as `runs/thermoml_crystal_cp_baselines_development_20260829_infrastructure_failure_quadratic_perturbation_lookup_20260829_0740/`.  The audit was changed only to compare each prediction with its support-only recomputation at generation time; the two fixture tests then passed in about eight seconds.  A clean absent formal root completed in `109.6 s`.  Final runner/result/manifest/prediction SHA-256 values are `66c80056a31907ef5a4b9ab84bf65770bef50ee27ba088d92c0cd311787ed6d7`, `bae3b633a8cbb98b050e04f3b40b59017796946a42e20ead1a136eef141d30da`, `54f9b384ff57064778ad9131134b4cd284f53ce6e4a5623b77295783880b7d6f`, and `719cdae0d651e9785fe8ac4385606055755fc9e710b21224df165627d5850feb`.  The spread-selected expression is Shomate5 with lambda zero; the same package, rather than an oracle re-selection, is reused in prefix/four-support stress.  Query-target perturbation changes candidate predictions/coefficients by exactly zero.  The independent analyzer is now terminal; its script/decision/analysis-manifest SHA-256 values are `9e5b1fe11c6b919064f81f966f293e7c0623dec59c475cb31695a881fcce99e3`, `a90390db591b0e5a10086ac35f67ce4b015d4a555f86d5a6ea46a19b67152638`, and `61f9535c67f19d943fb4c056eef41300d0dccc81945403f20ae397aa2f659de1`; two analyzer tests pass and the readable report is `runs/thermoml_crystal_cp_baselines_development_20260829/analysis/BASELINE_RESULTS.md`.

Independent analysis shows the central scientific split.  The v2 spread Shomate expression has physical pooled R2 `0.6386459473`, so the user's pooled `0.85` expression endpoint fails in this exact version, even though median entity NRMSE is `0.02958` and 238/247 entities have individual R2 at least 0.85.  Same-support linear and PCHIP reach pooled R2 `0.95208/0.95881`; PCHIP median entity NRMSE is `0.00872`.  The Shomate failure is not broad: one entity contributes `88.56%` of its SSE and the top three contribute about `99.95%`.  Entity- and DOI-bootstrap 95% intervals for Shomate-minus-PCHIP pooled R2 are respectively `[-0.4240,-0.000156]` and `[-0.4375,-0.000195]`, and Shomate beats PCHIP on only 38/247 spread entities.  The dominant curves are Crystal-labelled upper-bound anomalies: SRS-1,2,3,4,5-pentanepentol rises to `57,525`, 1,4-diphenyl-2,3-dithiabutane to `22,786`, and methyl butanoate has a narrow `8,659.7` peak.  The first two expose a strong positive final support secant, suggesting an effective transition-stage term rather than dataset exclusion.  Prefix Shomate has one visible entity coverage failure and the unchanged five-term lambda-zero Shomate fit has zero valid four-support predictions; neither is imputed or hidden.

A bounded development exploration tested normalized upper-bound log/pole atoms added to low-degree smooth backgrounds, without opening confirmation.  A support-only router based on the last positive secant divided by earlier median absolute secants plus a quadratic inverse-square-root boundary atom raised direct full-development pooled R2 to about `0.985` while retaining median NRMSE about `0.0297`; this was exploratory and is not confirmatory.  Before any formal nested result, `THERMOML_CRYSTAL_CP_TRANSITION_STRUCTURE_AMENDMENT_20260829.md` was frozen at SHA-256 `ee10462ad90d748d6f255a219ef2a05e222a19f545372ef3925c12c5aa018eee`.  It binds the complete explored gamma/degree/log-or-pole/delta grid, exact support-only stage ratio, SVD fitting, five outer DOI folds, constrained training-only selection, OOF gates, all-development final package, honest failure rules and the stage-wise interpretation `smooth lattice/background + effective near-boundary transition anomaly`.  No entity may be removed; confirmation remains single-use and unopened.

## 108. Crystal-Cp transition v3 fails narrowly; maximum-margin v4 frozen (2026-08-29 08:41 CST)

The transition runner was independently red-teamed before scientific execution.  The audit found no confirmation access or fold-target leakage, but required an independent analyzer and exposed missing critical tests.  During the first main-agent launch, the script failed before data access or formal-root creation because the repository root was not on `sys.path`; only the import boundary was repaired.  The expanded suite then exposed and fixed a real pre-result gate bug: the runner had compared the number of per-entity metric rows with the expected number of query points.  The final runner/analyzer tests cover DOI isolation, the strict router boundary, the full 1% tie-break, v2 hash/coverage/value mismatches, exact support roles/indices, raw-result recomputation and perturbation; 10/10 pass with an independent repository-local basetemp, compilation and `git diff --check`.

The exact frozen v3 formal root is terminal.  Runner/result/raw-decision SHA-256 values are `df48cb6674de34949f54cb768c7ff85dd03e606ffd3a28e7d578ca24609426e1`, `aec86dba8c9dc33f3942b473f08f8b2bf9479e84dcef496b1fe688af7e08ebaa`, and `8aa9ec409020d681bcbdceecd43e6fa39de337a687f7799f5070f761db817764`.  The independent analyzer, decision and manifest hashes are `807ba64be6db6cdf31f51b57a65dc2aabf0260ef7535d7fe2231c032b5681764`, `1d32cb2ed88509af08b21e81acf845f6811064d9fcf9bd4381c1bd337c8b5d13`, and `25d82c1c66735646db1ceacf55845598090310f516e841898f85e419569643db`.  Independent OOF pooled physical R2 is `0.8395842717`, so v3 fails the exact `0.85` endpoint; all other frozen gates pass, including exact 247/17,704 coverage, finite fits, median/p95 tails, individual-pass count, negative-prediction count, 4/5 effective motif recurrence and exact query-target invariance.  Entity/DOI bootstrap 95% intervals are `[0.75495,0.99976]` and `[0.72208,0.99976]`.  Confirmation remains unopened.

The failure is localized to a risk-equivalent threshold tie.  In outer fold zero, gamma 200 and 500 route the same one training anomaly, but the old larger-gamma tie-break chooses 500; both strongest anomalies are held out and the second has support-only stage ratio `405.97`, so it is missed.  One post-v3 development exploration replaced only that last tie by a maximum empirical log-stage margin while retaining the full formula/grid/constraints.  It yielded pooled OOF R2 `0.8593003362`, 748 negative predictions, thresholds `200/100/100/100/50`, inverse-square-root atoms in every fold, degree one in fold zero and degree two elsewhere.  This is explicitly development-induced, not untouched evidence.

`THERMOML_CRYSTAL_CP_ROUTER_MARGIN_AMENDMENT_20260829.md`, SHA-256 `07194d41108d177405d63682135dd9f1bbf2e419d7d72894dfcf81d4ee4920ae`, now freezes the single v4 change: among otherwise tied candidates choose gamma maximizing `min_i |log(1+r_i)-log(1+gamma)|`, then larger gamma on an exact tie.  Before any formal v4 result, a symmetric synthetic test exposed last-bit float64 inequality between mathematically equal log margins; the amendment now treats absolute margin differences at most `1e-12` as the exact numerical tie.  This portability clarification does not change any declared real-data selection.  The all-development package is declared in advance as gamma 100, degree-two inverse-square-root, delta 0.0003.  A formal v4 run may only reproduce/audit this development result; the unchanged one-shot 2015--2019 cohort is the decisive generalization evidence.  No further symbolic-family change is allowed after v4 confirmation.

Before any crystal-Cp neural result, `THERMOML_CRYSTAL_CP_NEURAL_IMPLEMENTATION_CLARIFICATION_20260829.md` was frozen and later expanded pre-run after an independent static red-team; its current SHA-256 is `7e04a98a50f381da296e96c93ebc1c717dd80b12687b35b53695ef2985343aff`.  It fixes HSIC plus `rich_rff_kme` for the 0.05 orthogonality term, retains minimum-norm stable-GN diagnostic predictions for rank-deficient Jacobians while classifying them as jointly unidentified, and forbids formal neural launch until its decoder-functional/support re-q basis binds the final eligible symbolic package.  The pre-run expansion applies the same stage-ratio formula to four supports using the last slope versus the median of the first two, treats absent/insufficient finite-lambda votes as `NOT_TESTED` rather than a cell crash, requires typed hashes of the v4 amendment/analysis/final package/fold selections, and assigns cross-seed/support-offset OMP certificates to the aggregate analyzer.  No crystal-Cp neural smoke or formal result existed before either clarification.  A host-level read-only GPU refresh showed cards 2--5 empty at 4 MiB with no compute processes; cards 0/1/6/7 were occupied by other users and were not touched.

## 109. Crystal-Cp maximum-margin v4 passes the development expression gate (2026-08-29 08:58 CST)

Before formal v4 execution, a symmetric synthetic test found that two mathematically equal log margins differed in the final float64 bit.  The protocol was clarified pre-result to treat absolute margin differences at most `1e-12` as an exact numerical tie and then choose larger gamma.  The amendment's final SHA-256 is `07194d41108d177405d63682135dd9f1bbf2e419d7d72894dfcf81d4ee4920ae`; this portability rule changed none of the declared real-data selections.  Four v4 runner/analyzer tests, compilation and `git diff --check` passed, and the exact formal root was absent before launch.

The v4 formal reproduction and independent analyzer are terminal.  Runner/analyzer/result/raw-decision/analysis-decision/analysis-manifest SHA-256 values are `3fa244e55a2238aad238f63b16f3c725e538853885f9efe6c1bc4e68907fc9d2`, `f08aa890aef8e7043f41adc0115b1facc396712b74a2899aee40449e5c7fbb47`, `6b29fc84fcfc67270252f53c10a1c8993c56bebef95f4186db2e381fb7c22a39`, `81f6d834e999cc18e339649946c2e235a45ecf2e1194d0de8ecc8ae1c52aac77`, `09c9cb17cf64cc5e6f5bc3b9958ed07ed7fda56245b545961c4312647503e0e8`, and `6c023c8ba342c0244e9680a0fc88f4b94001d8cf326d77330c3eea5e5458c1a0`.  All raw and analysis manifest file hashes reconcile exactly.

Independent development OOF physical pooled R2 is `0.8593003362`; all ten frozen gates pass with exact 247-entity/17,704-query coverage, finite fits, 748 negative predictions, unchanged median/p95 and individual-pass constraints, 4/5 degree-two inverse-square-root recurrence, and exact million-unit query-target invariance.  The fold thresholds reproduce exactly as `200/100/100/100/50`; the all-development sealed structure is gamma 100, degree two, inverse-square-root atom, delta `0.0003`.  Entity/DOI bootstrap 95% intervals are `[0.82526,0.99976]` and `[0.81641,0.99976]`, so the point endpoint passes but uncertainty does not justify replacing the still-required untouched temporal confirmation.  Confirmation remains unopened.

The intended stage-wise expression is now concrete: ordinary entities use the smooth Shomate background; support-detected boundary-stage entities use `Cp=A+B1*u+B2*u^2+H/sqrt(delta+1-u)`, `u=(T-Tmin)/(Tmax-Tmin)`, with shared `delta=0.0003`.  The polynomial is a smooth lattice/background coordinate; `H` is an effective near-boundary anomaly amplitude and the implied boundary is `Tc=Tmax+delta*(Tmax-Tmin)`.  It is a stage-wise explanatory coordinate, not a unique critical law or raw-q recovery.

## 110. Crystal-Cp train-only FPCA is terminal and does not explain the transition result (2026-08-29 09:12 CST)

The frozen `FPCA_support_ridge` comparator is now terminal on the complete development cohort without opening confirmation.  For each outer DOI fold it fits a 101-point normalized-temperature PCA basis only on complete outer-training curves; held-out coefficients use only the declared support targets.  The component grid is `2/3/4/5/8`, the ridge grid is `0/1e-6/1e-4/1e-2/1`, exact query coverage is `17,704/17,897/22,754` for spread/prefix/four-support, and query-target perturbation changes predictions by exactly zero.  Two focused tests, compilation, diff-check, the formal run and an independent raw-output analyzer all passed.

The selected spread configuration is three components with ridge `1e-4`; pooled physical R2 is `0.6321885368`, median entity NRMSE `0.355030`, p95 `9.685081`, and 128/247 entities reach individual R2 at least `0.85`.  Prefix reaches pooled R2 `0.2864901930`; four-support reaches `-1.3877600725` despite a lower median entity NRMSE `0.170674`, because the pooled metric is dominated by large boundary failures.  Thus a generic train-only low-rank functional coordinate does not reproduce the v4 transition expression's `0.8593003362` pooled result.  This is a useful positive control for the scientific story: the compact support-routed boundary term captures structure missed by unconstrained smooth FPCA, while FPCA remains a non-symbolic comparator and its failure is not an expression claim.

Runner/analyzer/raw-manifest/analysis-decision/analysis-manifest SHA-256 values are `1e7a278df8221a9212c988d391664694b30e42cf8ab14e0c0e70bcc67367f72f`, `170a4e7746d4326b312fa73c0388fb1d519bda3a10286188d86c2a2931facf8f`, `31cd416618896f08e99168e284d67094f0fc998f8e95cec64e100e42ca3d81f3`, `ea03cf8feaeb4aa118d934484b25eb86ed061267a2a4cb63b09c79ed3397b28e`, and `f21899226e3d36daf2bcbb4564fee7148c91e1907cd718be7a0aa1e40c27df6e`.  The readable result is `runs/thermoml_crystal_cp_fpca_development_20260829/analysis/FPCA_RESULTS.md`.

## 111. Crystal-Cp matched no-q formal launch (2026-08-29 09:14 CST)

The matched `no_q_temperature_mlp` execution chain was independently implemented and then corrected before formal launch after main review found that the first draft scored only the spread regime.  The final runner trains once per fold/seed and performs one held-out full-curve inference, then writes the exact spread/prefix/four-support query views with coverage `17,704/17,897/22,754`.  Each regime has an independent copied-data `+1,000,000` query-target reload probe.  The analyzer requires five seed predictions for every `(regime,source_row_id)`, computes the pointwise median separately by regime, and reports entity/DOI bootstrap, tails and support-range strata.  The architecture/budget remains `(256,128)`, 1,000 epochs, Adam `1e-3`, batch 256, one full deterministic training-row pass per epoch and inverse-entity-row-count weighted physical-target MSE after outer-train-only standardization.  It has no entity ID, support, q or query-target input.

Runner/launcher/analyzer SHA-256 values are `5b2d919be05c97fccc8c83474970dd5fa934ca45a1fe21a589fc9b08f3c2db0b`, `88d0ac810cb90c3207acb568241195a549a6d307b733d6cbdd2c52529b71ac74`, and `dff9f69eec2b3eafde22bb08ce87214ffaa0dd7a090c5646bbbb3adbd45b0aa7`.  Main-agent compilation/diff-check and all three focused tests passed.  Immediately before launch, host GPUs 2--5 were each at 4 MiB with no compute process; 0/1/6/7 were occupied by other users and untouched.  Formal tmux `crystal_cp_noq_20260829` is active at `runs/thermoml_crystal_cp_no_q_temperature_mlp_development_20260829/`: first observation `0 completed / 4 active / 21 pending / 0 failed`.  Confirmation remains unopened.  Do not report partial scores as terminal evidence.

## 112. Expression endpoint reaffirmed independently of raw q and predictive superiority (2026-08-29 09:38 CST)

The user reconfirmed the paper's primary interpretability endpoint: a compact expression is successful when strict unseen-entity/support-query evaluation in physical units reaches pooled `R2 >= 0.85` and the expression offers a credible stage-wise scientific clue. It need not recover the initially learned raw `q`, a preselected physical variable, or a unique true law. Decoder-functional coordinates, structure-recalibrated q, and other response-aligned low-dimensional coordinates are eligible.

This clarification changes no frozen threshold or already sealed protocol; it fixes claim hierarchy. Starry ZT has already passed this endpoint in an untouched temporal cohort. Crystal-Cp v4 passes it on development OOF data (`R2=0.8593003362`) through the readable smooth-background plus near-boundary-anomaly structure, but its untouched temporal confirmation remains mandatory before claiming external generalization. Beating PCHIP/kNN/CNP/FPCA/no-q MLP, recovering the original raw-q gauge, and identifying a unique mechanistic law are separate stronger endpoints. Failure of one of those stronger endpoints must not erase a valid expression result, while an expression pass must not be presented as predictive superiority.

## 113. Crystal-Cp matched no-q terminal analysis (2026-08-29 09:43 CST)

The formal no-q chain is terminal at 25/25 successful CUDA cells with zero failures and exact spread/prefix/four-support query coverage `17,704/17,897/22,754`. The independent analyzer passes every integrity gate and confirms exact query-target perturbation invariance. Five-seed pointwise-median pooled physical R2 is `0.0957131964` for spread, `0.0965977177` for prefix, and `0.1220088881` for four-support. Spread median/p95 entity NRMSE is `3.583923/39.649466`, and only 14/247 entities reach individual R2 at least `0.85`. The matched temperature-only MLP therefore does not explain the transition expression's development OOF `0.8593003362`; entity-specific support information is essential.

The first independent analyzer invocation stopped before creating its analysis root because CUDA-saved float32 predictions and CPU checkpoint replay differed at backend rounding level. The final audit compares outer-train-standardized predictions with `rtol=1e-6, atol=5e-6`, records the maximum standardized difference `3.5762787e-6` and maximum physical difference `0.00343035`, and retains exact artifact hashes. This is a reproducibility-tolerance clarification, not a scientific-result change. Analyzer/decision/analysis-manifest SHA-256 values are `df6f4a9d380a1f723550c290ead37b14992b52b4711ed684f357992bfef002c2`, `6f2bd61f82f20cfa843e600c779a912fbdb595cbaab9d388118f41f27e7b845a`, and `f281e39e65e11aa863747f89d052b4befb8f013f156db3c299b9ef01f805afd1`. The readable result is `runs/thermoml_crystal_cp_no_q_temperature_mlp_development_20260829/analysis/NO_Q_TEMPERATURE_MLP_ANALYSIS.md`. Confirmation remains unopened.

## 114. Crystal-Cp support-CNP formal launch (2026-08-29 09:49 CST)

The 25-cell `support_cnp` matrix is formally active on physical GPUs 2--5 at `runs/thermoml_crystal_cp_support_cnp_development_20260829/`. Immediately before the successful launch, host-level `nvidia-smi` showed all four selected cards at 4 MiB, 0% utilization and no compute PID; cards 0/1/6/7 were occupied by other users and untouched. The first live state is `0 completed / 4 running / 21 pending / 0 failures`, with fold0 seeds0--3 on GPUs 2--5 and about 717 MiB observed per card. The isolated host-visible tmux socket is `lvs_crystal_cp_cnp_20260829`, session `crystal_cp_cnp_20260829`. Confirmation remains unopened and no partial score is a terminal claim.

Two pre-training infrastructure failures are preserved. The first reused a default tmux server without NVIDIA device visibility and created only an empty `logs/` directory. The second host-visible launch exposed a shared-machine guard bug: compute processes on unrequested GPUs 0/1/6/7 were mapped into a PID dictionary containing only requested GPUs 2--5, raising `KeyError('0')` before any cell. Their empty roots were moved to `runs/thermoml_crystal_cp_support_cnp_development_20260829_infrastructure_failure_tmux_gpu_namespace_20260829_0945/` and `runs/thermoml_crystal_cp_support_cnp_development_20260829_infrastructure_failure_foreign_gpu_pid_keyerror_20260829_0948/`. The launcher now ignores process rows outside the requested GPU set; the existing scheduling test includes a foreign busy GPU and the full focused suite passes 4/4. Current launcher/test SHA-256 values are `56553eab89f6ae9404f177850d5a4dc50193e33d128646abd3a51de42f4a2609` and `9bd9e0699f3fe733bb9d2474640af789f8d7749397537315064225b728596911`.

## 115. Crystal-Cp neural/GIRD real-data smoke passes before formal launch (2026-08-29 09:53 CST)

A repository-local, non-counted fold0/seed0 CPU smoke completed successfully at `runs/_smoke_thermoml_crystal_cp_neural_gird_real_20260829/`. It used two epochs, three Adam calibration steps, two stable-GN steps and one affine gauge; `scientific_selection_eligible=false`. The cell produced every expected checkpoint, q, support/query, calibration-path, decoder-probe, canonical-coordinate, dictionary, GIRD, gauge and prediction artifact. It records 203 outer-training and 44 outer-test entities, 150 optimizer/q/backward steps, 38,004 processed examples, 163,422 prediction rows, global-regularizer scale sum `1.0000000000000009`, zero query-target perturbation difference, and `four_support_conditional_status=READY_FOR_ANALYSIS` with four contributing lambda folds. Training/calibration wall times were `5.25/22.70 s` on CPU; these shortened timings and all smoke metrics are plumbing evidence only. The formal neural/GIRD root remains absent and must not be launched until the active support-CNP matrix releases GPUs 2--5 and availability is rechecked.

## 116. Crystal-Cp support-CNP first transition (2026-08-29 09:58 CST)

The formal support-CNP controller has completed its first four of 25 cells with zero failures and dispatched the next four, leaving 17 pending. Fold0 seeds 0--3 each finished 1,000 epochs with exactly 13,000 optimizer updates and 13,000 backward calls; their training wall times were 523--531 seconds. Host-visible GPUs 2--5 remain occupied only by the next CNP cells at about 821--823 MiB each and 10--12% utilization. Cards 0/1/6/7 remain occupied by unrelated users and were not touched.

The four early fold0 spread pooled physical R2 values are `0.08324/0.06762/0.06194/0.07668`; prefix values are about `0.013--0.015`, and four-support values are negative. All query-target perturbation differences are exactly zero. These are partial, single-fold per-seed observations and must not be promoted to a terminal comparison; the independent analyzer remains gated on 25/25 completion and pointwise five-seed aggregation.

## 117. Crystal-Cp confirmation transaction design and CNP second transition (2026-08-29 10:08 CST)

The support-CNP controller has now completed 8/25 cells, keeps four active, has 13 pending and zero failures. GPU/process observations remain healthy and each cell takes about 8.8 minutes. No partial score has been promoted.

A read-only design review of the eventual Crystal-Cp confirmation established a mandatory pre-confirmation step that was not yet implemented: OOF fold checkpoints cannot be reused as temporal-confirmation models. After CNP and neural/GIRD development analyses become terminal, five-seed all-development CNP, auto-decoder/GIRD and no-q packages plus the selected FPCA/dictionaries must be generated and sealed. The confirmation runner must then follow the already proven ThermoML transaction pattern: verify the seal and code hashes; derive a unique root from the seal hash; atomically write and fsync an exclusive consumed lock and first hash-chain receipt before opening any selected response; materialize exactly 86 entities/62 DOI/11,447 rows; build separate redacted inference views for spread/prefix/four-support; run separate million-unit perturbation probes; and append either a terminal-success or failed-and-consumed receipt. A failed first attempt is still consumed and cannot be retried.

Metadata-only confirmation coverage is already fixed without response access: spread support/query `2,890/8,557`, prefix `2,829/8,618`, and four-support `344/11,103`. The v4 transition expression is eligible only for the frozen spread endpoint because its stage detector requires at least five support points; it must not be retrofitted to the four-support regime. Final confirmation code, method registry, all-development checkpoints and their hashes cannot be sealed until CNP and neural/GIRD development are terminal.

## 118. Canonical-response claim gate supersedes the old MATR-centered gate (2026-08-29 10:16 CST)

`ICLR_CANONICAL_RESPONSE_CLAIM_GATE_20260829.md` is now the authoritative claim hierarchy and ICLR stopping rule. The older `ICLR_CLAIM_EVIDENCE_GATE_20260828.md` is intentionally retained as a historical record rather than silently overwritten. The new gate separates support-state evidence, affine-gauge/canonicalization evidence, the user-defined interpretable-expression endpoint, and the independent stronger predictive-superiority endpoint. It records Starry ZT and ThermoML vapor-pressure temporal confirmations, Crystal-Cp v4 development evidence and uncertainty, the pending CNP/neural/confirmation gaps, prohibited title/abstract claims, and the exact transactional confirmation requirements. The main report links this gate from its leading 2026-08-29 narrative notice.

At the same timestamp, formal support-CNP progressed to 12/25 completed, four running, nine pending and zero failures. No partial score was promoted and confirmation remains unopened.

## 119. CNP checkpoint replay tolerance audited before terminal analysis (2026-08-29 10:21 CST)

A read-only CPU replay of the completed fold0/seed0 CUDA checkpoint exposed that the original analyzer's physical-unit absolute tolerance `1e-5` was not a valid backend-reproduction gate. The CPU/CUDA maximum differences in outer-train-standardized target units were only `4.77e-7/4.77e-7/5.72e-6` for spread/prefix/four-support, while multiplication by the physical target scale produced maximum physical differences up to about `0.00332`. The previous gate would therefore have rejected an identical checkpoint solely because of float32 backend rounding.

The independent analyzer now compares replay in the exact standardized coordinate optimized during training with fixed `rtol=1e-6, atol=1e-5`, and records both the maximum standardized and maximum physical differences in the formal decision. This changes no model, prediction, metric, method selection or scientific gate. A focused regression test verifies that a `6e-6` standardized discrepancy passes even when the physical scale amplifies it above `1e-3`, while a `2e-5` discrepancy fails. The complete focused suite is `5 passed`; compilation and `git diff --check` pass. Current analyzer/test SHA-256 values are `a0837b29f9aace0d0a38fcc58eb71fcc3c9af9374599eb1bb08e628a5a05cacc` and `49fab1a774caec15395a40b084ded66e41f3116cd8e58b567a8d33fbb43a1aa6`.

## 120. Support-CNP terminal analysis and neural/GIRD formal launch (2026-08-29 11:00 CST)

The formal support-CNP matrix reached 25/25 successful cells with zero failures. All cells used the frozen DeepSets `(128,128)` support encoder, `(256,128)` query head, 1,000 epochs, exact entity batches and episode schedule; total training time is 13,434.45 seconds across cells and total optimizer/backward calls are both 320,000. Confirmation remained unopened.

The independent analyzer required two additional non-scientific repairs before it could run. First, its exact query coverage gate compared equal ID sets in CSV order rather than as sorted multisets. Second, it mechanically constructed `four_support_role`, although the sealed column is `four_role`. Both failures occurred before `analysis/` creation. The analyzer now uses an explicit frozen regime-to-role mapping and sorted exact multiset comparison; duplicate IDs still fail. Regression tests cover reordered IDs, duplicate IDs and the exact three-column mapping. Together with the standardized replay audit, the final focused suite is 6/6 passed. Final analyzer/test SHA-256 values are `d6a3f438d7875c5beab843ae359ce1093db2736b64ef282b03aa07b4cddba7a2` and `576e3b49b4f4e814ae09b7fbd8a3b3cdb0f940ca9b48dc97bf396bc81b94892a`; decision/analysis-manifest hashes are `0fefb4afc47aabd5ca14dd38fca7465184f129925fe570bb51a11ce4568bcc80` and `19cfcca06d11b38a6a40c264fd11c3793f09fbb6603a8c7b3991a2f6d09dcae9`.

All CNP integrity gates pass with exact spread/prefix/four-support query coverage `17,704/17,897/22,754`, five predictions per query point, finite predictions, 247 entities, exact query-target invariance and checkpoint replay. Five-seed pointwise-median pooled physical R2 is `0.3464951730` for spread, `0.2744412333` for prefix and `0.0464183028` for four-support. Spread median entity R2 is `0.867179`, 126/247 entities reach individual R2 at least 0.85, median/p95/max NRMSE is `0.364446/2.840361/20.590272`, and 1,375 predictions are negative. Entity/DOI bootstrap pooled-R2 intervals are `[0.161913,0.991570]` and `[0.136919,0.991619]`. Across entities the median five-seed prediction-rank Spearman is 1.0 but the 5th percentile is about 0.649; within-seed support-offset prediction-rank Spearman median/5th percentile is `1.0/0.99756`, with median offset-pair RMSE `1.7767` physical units.

Durable interpretation: learned support conditioning is substantially stronger than the support-blind no-q MLP (`0.3465` versus `0.0957`) but remains far below the v4 stage expression (`0.8593`) and local PCHIP/linear/kNN. The gap is concentrated in high-amplitude/tail entities. This closes the missing fair learned support-aware baseline; it does not by itself prove that explicit raw q is necessary. The main report and `ICLR_CANONICAL_RESPONSE_CLAIM_GATE_20260829.md` now include the terminal CNP result.

Immediately after CNP released the cards, host `nvidia-smi` showed GPUs 2--5 at 4 MiB with no compute PID. Fourteen neural/GIRD focused tests, compilation and all frozen runner/launcher/analyzer/adapter hashes passed; its formal root was absent. The 25-cell neural/GIRD matrix was then launched on GPUs 2--5 in isolated tmux socket `lvs_crystal_cp_neural_gird_20260829`, session `crystal_cp_neural_gird_20260829`. First state is four active fold0 seeds, 21 pending, zero failures; each PID established a CUDA context using about 698 MiB. Runner/launcher/analyzer/adapter hashes are `d296d75b56dc5857cba8c4c53d25ed610d57e7cb91f0622154ce2e44a3c7d723`, `13fbbd6bc4136c7226e037f95d438ed3e0a4ab184f745e00bbe34e61819589ba`, `b1bb7826bcaf003c37823c4f243e6360856a6eef677fbe02a782d0784a0bd210` and `0b402d6c6b7fe7b474e11046ec0c158e12bf1f5def5695d4bc0658119a30ee80`.

## 121. Expression endpoint and Crystal-Cp confirmation contract review (2026-08-29 11:10 CST)

The user again fixed the durable success criterion: a compact symbolic expression is interpretable enough for the paper when strict unseen-entity support-to-query evaluation reaches physical pooled `R2 >= 0.85` and its named terms provide a credible stage-wise scientific clue. The coordinate may be decoder-functional, support-recalibrated, or otherwise response-aligned. It need not equal the first learned raw `q`, a preselected physical quantity, a unique law, or the true microscopic mechanism. Predictive superiority over PCHIP/kNN/CNP/FPCA/no-q remains a separate stronger endpoint and cannot be inferred from this criterion.

`THERMOML_CRYSTAL_CP_SINGLE_USE_CONFIRMATION_CONTRACT_20260829.md` now freezes the transaction boundary, exact 86-entity/62-DOI/11,447-row cohort, three metadata-only support/query regimes, redaction and million-unit perturbation tests, receipt-before-target-access order, failure-consumes-once rule, and spread-only eligibility of v4. Main-agent review found and corrected two draft inconsistencies before acceptance: the comparison package now states five sealed neural seeds rather than three, and the primary expression gate requires finite complete predictions rather than all-positive predictions. Negative predictions remain mandatory diagnostics; they cannot be used to remove entities or change the frozen `R2 >= 0.85` point gate. The synthetic contract suite passes 3/3 and opens no confirmation response or formal confirmation root.

At the same timestamp, the neural/GIRD formal run remained healthy at 0/25 terminal cells, four active fold0 seeds, 21 pending and zero failures. GPUs 2--5 each used about 799 MiB at 31--35% utilization; the four cell processes each used about 97% CPU after roughly 15 minutes. No partial metrics were promoted and confirmation remained unopened.

## 122. All-development CNP packager accepted and paired v4/CNP tail advantage quantified (2026-08-29 11:16 CST)

The independent all-development support-CNP packager now trains five seeds on all 247 development entities/159 DOI/23,742 rows with the exact formal DeepSets `(128,128)` encoder, `(256,128)` query head, three-regime episode schedule, 1,000 epochs and 16,000 optimizer/backward calls per seed. It never references confirmation selection/source/response paths. Main-agent review strengthened `verify_package`: it now validates formal coverage and budget, every seed's provenance and terminal summary, every nested artifact hash, reconstructs the CNP, loads the checkpoint, checks finite weights and verifies checkpoint/JSON normalizer equality. A nested-training-history tamper regression was added. The component package explicitly sets `authorize_confirmation_evaluator=false` and requires a later joint component/code seal. The final support-CNP packager/test SHA-256 values are `410636116d6de7ba1beaf6816c9a981f477d86bdc68aa903ec1741bbad39e7a9` and `61e05a7a672940b9e4ac1b22973a0a5cda19a126b28ad4bb7c2c51eed4240c0c`. Formal all-development training has not started and must wait for the current neural/GIRD development matrix to release or share the authorized GPUs safely.

A source-row-exact read-only pairing of terminal development artifacts adds an important descriptive comparison. The CNP spread median file SHA-256 is `670b5c863452316bb49ba222e45fdae880f023c19cb5c2810fd0215d2d98c029`; the v4 OOF point file SHA-256 is `02234fe74379d02a94a645e92016666a50f7546719d0b3002e78c5a6190ea122`. After filtering v4 to `method=transition_selected`, both contain the same unique 17,704 query `source_row_id` values and identical targets. Using each entity's query-target standard deviation as the common NRMSE denominator, v4 is lower on 243/247 entities (`98.38%`); after grouping entity NRMSE by DOI median, v4 is lower on 156/159 DOI (`98.11%`). CNP versus v4 median NRMSE is `0.364446/0.029578`, p95 is `2.840361/0.211864`, and maximum is `20.590272/1.744952`. This supports a broad paired tail advantage over the generic learned support encoder, not predictive superiority over PCHIP/linear/kNN and not raw-q necessity. Confirmation remains unopened.

## 123. All-development neural/GIRD trainer and component registry accepted (2026-08-29 11:23 CST)

Development-only infrastructure now exists for the eventual five-seed all-development auto-decoder/GIRD refit. Its contract freezes all 247 entities/23,742 rows, q=4, `(256,128)`, 1,000 epochs, label-balanced physical MSE, q-L2/HSIC/continuity regularizers, stable GN, 41 probes, OMP/rank/condition rules and the sealed v4 basis adapter. The formal trainer fails before creating its output root until a new `analysis/all_development_decision.json` binds the terminal 25-cell launcher, independent aggregate decision and manifest. It has no defaults for per-regime decoder/direct-target K, spread/prefix lambda, four-support conditional state or claim status; none may be inferred from partial cells or smoke data. After K is frozen, atom identity is deterministically recomputed on all development entities and is an artifact, not another selection.

Each seed will save a loadable model/embedding checkpoint, label map, normalizer, q prior, four Hadamard starts, stable-GN q/path/Jacobian/probes, decoder/direct-target dictionary inputs, complete OMP paths and selected dictionaries. The packager verifies every nested hash and produces a pointwise-five-seed registry for raw auto-decoder, decoder-functional, support structure re-q, all fixed-lambda GIRD endpoints, selected/conditional GIRD, support OMP, decoder-only and direct-target dictionaries. The component seal explicitly keeps `authorize_confirmation_evaluator=false`; a later joint baseline/code seal is still required.

Main-agent review added three reproducibility gates before acceptance: formal Torch inter-op threads now match the 25-cell runner; every all-development seed must have exactly `1000*ceil(23742/256)=93,000` theta steps, q steps and backward passes plus 23,742,000 processed examples; and package verification binds manifest/checkpoint seed identity. A terminal ledger was added per seed. After the terminal-decision review described in section 126, the final focused and compatibility suite passes 26/26, compilation and diff-check pass, and both neural/GIRD and CNP formal all-development roots remain absent. Current plan/preparer/trainer/packager/preparer-test/trainer-test SHA-256 values are `7351d585d4e6ad17ae8686d1d41b8397586bcb8c99803b6d2238839c14ef012f`, `1c66eed8401f33997e5914cb3b2ce60b27acd06bd690a5ca1c0365881286ed80`, `0d5f5f2f173f2287292d26171fad34720447edb151346c8b4e42e65f22ec3b58`, `ed8368bcad31ef77629b61dbc89099d7157d3edf4432fe520bc50e561523d508`, `19460d6370c4ca0728454f960c79bdceecdabe7444a2d643e6b3efb197f463ae` and `d80c0fb98629acaa57c0c6c0f158d2b9d7de65b4349a15684d09778afe801100`. At this timestamp the formal development matrix remained 0 completed, four active, 21 pending and zero failures; confirmation remained unopened.

## 124. All-development matched no-q component accepted (2026-08-29 11:29 CST)

The all-development matched no-q packager now trains five temperature-only `(256,128)` seeds on all 247 development entities/159 DOI/23,742 rows for 1,000 epochs, Adam `1e-3`, batch 256 and inverse-entity-row-count weighted standardized physical-target MSE. Each formal seed must make exactly 93,000 optimizer/backward updates. It records an all-development-only normalizer, checkpoint, history, summary, terminal ledger, seed manifest and package-level nested hash inventory. It has no entity ID, support, q or query-target input and never references confirmation selection/source/response.

Main-agent review corrected the initially untested handoff by using the existing `.venv-lvs-gpu` environment; the combined all-development/formal no-q and CNP suite passes 16/16, compilation and diff-check pass. The no-q checkpoint now preserves the exact outer-runner top-level temperature/target normalizer fields as well as the structured normalizer, and package verification actually reconstructs the MLP, loads the state dict, checks finite weights and verifies checkpoint/JSON normalizer equality. Both no-q and CNP component packages explicitly keep `authorize_confirmation_evaluator=false`. Final no-q packager/test SHA-256 values are `7d00b8c2a6b95bd222ae0e2f0f34b23d08629afdd501db013f65c7db091768f1` and `76c82f82ee003f8c7d414930d22c007db5410740ff96cd03b1de6380851a14b2`. Its formal root remains absent and confirmation remains unopened.

One infrastructure warning is external to the project: `/tmp` briefly reached 100% inode usage and caused an `apply_patch` sandbox-mount failure even though project pytest temp/caches stayed repository-local. No unrelated `/tmp` material was removed or moved. The inode pressure later dropped to 96% without intervention, and all project verification completed under `runs/_runtime_cache`.

## 125. GIRD algorithm box and all-development FPCA component accepted (2026-08-29 11:38 CST)

`LATENT_Q_INTRO_MOTIVATION_THEORY_RELATED_WORK.md` now states GIRD as one reproducible algorithm rather than scattered diagnostics. Development cross-fits the decoder by entity/DOI, calibrates inner-heldout q in response space, selects a shared named dictionary by deterministic multi-response OMP, chooses K/lambda/rank gates only through nested development folds, then refits the decoder on all development entities after those degrees of freedom are frozen. At inference, an unseen entity's support yields a decoder-functional prior and a structure design in the same probe-Gram metric; the method falls back exactly to lambda zero when support identifies the dictionary, enables a frozen finite prior only when structure support is ill-conditioned and decoder support is identifiable, and keeps lambda zero/infinity, direct-target dictionary, FPCA, CNP, interpolation and no-q endpoints visible. Raw q is diagnostic rather than a named scientific output. This algorithm box separates decoder learning, readable coordinates and neural-prior incremental value, so failure of the stronger bridge cannot erase a valid external expression and an expression cannot impersonate a bridge pass.

The all-development FPCA component packager is independently ready. It verifies the exact terminal OOF raw manifest, selection, decision, analysis manifest, runner/analyzer and sealed data hashes, then refits the 101-point normalized-temperature basis on all 247 development entities/159 DOI/23,742 rows without any confirmation access. It copies the frozen OOF configurations without reselection: spread/prefix `(3 components, ridge 1e-4)` and four-support `(3, ridge 0)`. A main-agent read-only validation gives components shape `(101,101)`, basis rank 101 and minimum support ranks `3/2/3` for spread/prefix/four-support; prefix rank deficiency is retained and regularized rather than hidden. The component saves mean/components/grid/singular values/normalizer/config/rank diagnostics with nested hashes and keeps `authorize_confirmation_evaluator=false` pending the joint seal. Focused plus compatibility tests pass 6/6, compilation/diff-check pass, and the formal root remains absent. Packager/test SHA-256 values are `9a315966da454ec13d603eb469498e988779b20292cc1651618bc4c69c0eb40d` and `35f8942c86624493a31beab1f182f3a58b6dc49cef1939e6a0e6625a33d19166`. Confirmation remains unopened.

## 126. Pragmatic expression endpoint reaffirmed and terminal-decision semantics corrected (2026-08-29 11:46 CST)

The user explicitly reaffirmed the pragmatic paper endpoint: a compact expression with strict unseen-entity support-to-query physical pooled `R2 >= 0.85` is sufficiently interpretable when its terms offer a credible stage-wise clue. It does not need to equal the first/raw q, recover a preselected physical variable, identify a unique law, or be microscopically correct. This is already the authoritative endpoint in the canonical claim gate and main report. Predictor superiority remains a separate stronger endpoint, so a stronger PCHIP/kNN score cannot erase a valid interpretable-expression result, and an expression pass cannot be relabeled as predictive SOTA.

Before any terminal neural/GIRD aggregate or all-development output existed, main review found two implementation-only defects in the new decision extractor. The aggregate analyzer emits conditional scientific verdicts `PASS/FAIL/NOT_TESTED`, while cells emit evidence-availability states `READY_FOR_ANALYSIS/NOT_TESTED`; the first extractor draft incorrectly expected the cell vocabulary at aggregate level. The corrected frozen rule copies `PASS/FAIL/NOT_TESTED` into `claims.conditional_gird_status`, maps aggregate `PASS` or `FAIL` to deployment `READY_FOR_ANALYSIS`, and keeps `NOT_TESTED` undeployed. A regression test proves that a failed scientific claim still freezes the available comparator rather than deleting it. The extractor also now permits its already-existing formal `analysis/` parent while still requiring the output itself to be absent. No data, metric, selection, threshold or result changed.

The corrected selection-rule SHA-256 is `8de7ee18c6dcb9d4bf82a6bb928b613ddc909282e015d3efaef9981b4df6668d`. The four-file neural/decision compatibility suite plus cell/shell tests passes 26/26; compilation and `git diff --check` pass. The live neural/GIRD matrix remains `0 completed / 4 active / 21 pending / 0 failures`, with fold0 seeds 0--3 on GPUs 2--5 using 790 MiB per process at about 9--11% instantaneous GPU utilization. Other users' jobs on GPUs 0/1/6/7 remain untouched. The legacy extended campaign remains terminal at 2910/2910 with zero failures. Confirmation remains unopened, and no partial neural score has been promoted.

## 127. Formal all-development FPCA sealed and independent ICLR gap audit returned (2026-08-29 11:54 CST)

The formal all-development FPCA component is now terminal at `runs/thermoml_crystal_cp_all_development_fpca_20260829/`. It uses all 247 development entities/159 DOI/23,742 rows and copies the already frozen OOF choices without reselection: spread/prefix `(K=3,ridge=1e-4)` and four-support `(K=3,ridge=0)`. Independent `verify_package(..., expected_formal=True)` passes and the combined single-use-contract plus four-component package suite passes 27/27. Component-seal and manifest SHA-256 values are `e94291df93fd99118e9769cf0ac922d155011428658ca5907cd752daa188ed14` and `768a1cdc319b446fb8d5334f4fbaa154fa0e94d9bccd32d70cac99955417c193`. The seal retains `authorize_confirmation_evaluator=false`; confirmation remains unopened.

A fresh read-only ICLR gap audit, which opened no confirmation source and used no GPU, identified the two decisive remaining method-level risks even if Crystal-Cp temporal R2 passes 0.85. First, the strongest equations currently come from response projection or support-only structure re-q rather than a demonstrated end-to-end raw-q bridge; the active neural/GIRD terminal result must decide whether the learned decoder prior adds measurable value. Second, the theory is basis-relative affine-gauge canonicalization, not arbitrary nonlinear identifiability or automatic recovery of true physical variables. Other durable risks are lack of prediction superiority over PCHIP/kNN, formula families varying across case studies, and pooled R2 hiding entity tails/development-selection risk. Recommended main line: gauge-aware response-space canonicalization for support-conditioned scientific curves, with prediction SOTA, unique raw-q recovery, causal mechanism and universal law explicitly prohibited unless new evidence supports them.

## 128. Joint Crystal-Cp confirmation seal boundary frozen (2026-08-29 11:56 CST)

`THERMOML_CRYSTAL_CP_JOINT_CONFIRMATION_SEAL_PLAN_20260829.md` now freezes the final target-blind assembly boundary; SHA-256 is `200350b7276e975a84f7de234712ffe6fd8811b89430045118eac28086a44758`. The required inputs are the metadata-only 86-entity cohort seal and archive hash, v4 expression package/audit, terminal local-baseline selection/audit, five-seed CNP/no-q/neural-GIRD components, the already terminal FPCA component, and the final evaluator/analyzer/code/test/document hashes. It explicitly copies kNN sizes spread/prefix/four-support `2/1/2` from development without reselection. The fixed joint root remains absent; assembly cannot open an archive member or selected source JSON and cannot authorize a second attempt.

The plan keeps the expression and predictive endpoints separate, freezes the full method registry, and requires component-level `authorize_confirmation_evaluator=false` until one complete joint seal supersedes it. Only the joint seal may set `sole_authorized_confirmation_attempt=true`; its hash names the consumed root. The evaluator itself still must not be constructed until all 25 development cells and all-development components are terminal, as required by the single-use contract. At the latest host check the first four neural/GIRD cells had about one hour of continuously increasing CPU time at roughly 99% each, so they were progressing rather than stalled; state remained 0 completed, 4 active, 21 pending, 0 failures.

## 129. Guarded four-GPU all-development continuation is active (2026-08-29 12:04 CST)

A new minimal orchestrator, `scripts/launch_thermoml_crystal_cp_all_development_4gpu_20260829.py`, now binds the reviewed development analyzer/preparer, CNP/no-q/neural-GIRD/FPCA sources, exact formal roots and the existing GPU availability guard. It refuses to act unless development is cleanly terminal at exactly 25/25 with no active/pending/failures and no confirmation access. It then runs the frozen aggregate analyzer and all-development decision extractor if absent, verifies the terminal FPCA component, requires all four requested GPUs idle, and dispatches the CNP bundle, no-q bundle and five neural seeds without overwriting or retrying a failed task. After the seven GPU tasks succeed, it packages neural/GIRD and independently verifies the formal CNP, no-q and FPCA packages plus the exact neural seal.

The orchestrator and focused-test SHA-256 values are `65ef0af355579514c143cad86195f17ece85fb2fc40279661446477caada29d5` and `04737fa542b1a52975edb91815de7e650e94a8660b44af797e3deae30e671f7c`; the combined launcher/decision/component suite passes 30/30, compilation and diff-check pass. Host tmux socket/session `lvs_crystal_cp_all_dev_20260829:crystal_cp_all_dev_20260829` is active with `--wait-for-development`. While waiting it creates no campaign/component root and occupies no additional GPU; this was verified immediately after launch. On development `completed_all`, it will proceed only after the same reviewed source hashes and idle-GPU gate still pass. Its log is repository-local at `runs/_runtime_cache/launch_logs/crystal_cp_all_development_20260829.log`. Confirmation remains unopened.

## 130. Canonical-map invariance strengthened beyond affine charts without overclaiming calibration (2026-08-29 12:06 CST)

The theory now separates two scopes that the previous prose could invite reviewers to conflate. Proposition 1A defines probe-response equivalence directly: any two model/state representations that produce the same fixed physical probe response have exactly the same fixed-basis canonical coefficients, regardless of latent dimension, architecture, redundancy, invertibility or whether an explicit chart map exists. Thus the canonical map factors through the probe-response quotient; this representation-invariance theorem is not limited to affine gauges. It remains basis/probe relative and says nothing about causal or unique physical variables or response equality outside the probes.

The independently rerun support-to-q calibration algorithm has the narrower scope already established experimentally: global affine chart, full-rank support Jacobian, fixed response-space line search and stable QR/SVD Gauss--Newton. It is not claimed equivariant under arbitrary nonlinear test-time reparameterizations. The canonical claim gate and main report now state this split explicitly. Current theory/claim-gate/main-report SHA-256 values are `8dd6ded237b5dd33f3027dbdbb6301a65243424e28364a63e0a54f44f307c69b`, `721811d620ed00ecfdd5c27daef6455009efaaadef3f0ad75dbd2519a2ff62ce` and `6beaf4a8a4e1f452dbd039250615a3d678f7f6e969429de5f2b7bb3f1c35c9c8`. This addresses the ICLR audit's concern that the paper might accidentally present a conditionally affine calibration result as the entire invariance theorem, while preserving all empirical limitations.

## 131. Pragmatic endpoint integrated into the reviewed ICLR paper plan (2026-08-29 12:22 CST)

The user's durable criterion was integrated without changing its threshold: strict unseen-entity support-to-query physical pooled `R2 >= 0.85` plus a compact expression and credible stage-wise clue is the mandatory interpretability endpoint. It does not require the initial/raw q, a preselected physical variable, a unique true law or a microscopic mechanism. Decoder-functional and support structure-recalibrated coordinates are eligible. Predictive superiority remains a separate stronger endpoint.

The new root-level `PAPER_PLAN.md` and its timestamped source `PAPER_PLAN_20260829_121500.md` define the ICLR story as gauge-aware response-space canonicalization. The confirmed core branch is `support -> named basis -> stable structure re-q`; the optional learned-prior branch is `support -> calibrated decoder response -> canonical prior -> rank-aware GIRD`. Starry ZT and ThermoML vapor pressure are the two one-shot temporal confirmations. Crystal-Cp remains a pending third-domain stress test, and its terminal learned-prior increment is the go/no-go gate for promoting GIRD to a headline method contribution. The plan fits exactly 9.00 pages including title/abstract, limits the main body to three figures and two tables, and assigns all numbers to raw evidence roots.

An independent GPT-5.6-Sol xhigh outline review scored logical flow `8/10`, claim--evidence alignment `7/10`, experimental completeness `6/10`, prior-work positioning `5/10` and nine-page feasibility `4/10`, with a pre-revision Weak Reject/Borderline recommendation. The minimum fixes were applied: support re-q and neural-functional evidence are explicitly separated; the external functional `R2=0.9904` is diagnostic rather than the confirmed vapor-pressure expression; GIRD is optional until real incremental value passes; unverified priority wording was removed; the stage-wise clue requirement was operationalized as a named basis, coefficient stability and a falsifiable hypothesis; and the figure/table count was reduced. The reviewer judged the `R2 >= 0.85` expression-fidelity endpoint coherent and the prediction-SOTA separation clean. The remaining ICLR risk is algorithmic novelty if the frozen real decoder-prior increment is negative.

At 12:22 CST the authoritative neural/GIRD launcher JSON remained actively updated at `0 completed / 4 active / 21 pending / 0 failures`, with `confirmation_targets_opened=false`. This execution container could not access the NVIDIA driver or the host tmux namespace, so no GPU-occupancy claim was made from local `nvidia-smi`; the live JSON, raw results and guarded host orchestration remain the source of truth.

The initial reviewed paper-plan snapshot had SHA-256 `167ca32b1d7b3340f663f6323be09c3bb65f65d280bcd2329e5eb56ccb5ceb4e`; section 132 records the later literature-audited revision. The same operational staged-clue definition was synchronized to the canonical claim gate and beginner-readable main report; their pre-literature-audit SHA-256 values were `95cf545d922418fa5d774d467d44051a8840babec246ac1bb38d113cda0a1094` and `688d14ac21301e4529d8d8e6e98d2f61e972268cbebce7a44b9c479dadf23454`.

## 132. Verified closest-work boundary closes the paper-positioning gap (2026-08-29 12:33 CST)

`ICLR_CLOSEST_WORK_NOVELTY_MATRIX_20260829.md` now records a primary-source search across support-conditioned function prediction, implicit/function-level representations, latent identifiability, canonicalization and neural-to-symbolic equation discovery. It verifies CNP, MetaFun, Function Contrastive Learning, MetaSDF, Functa, Locatello, Khemakhem nonlinear ICA, Syrota metric structures, Ma canonicalization, Dym continuity/impossibility, Champion SINDy autoencoder, Cranmer neural-symbolic extraction, parametric equation discovery, UPINN and LASR. An independent Luna audit additionally surfaced the Khemakhem/Ma/Dym boundary, and main-agent verification used official PMLR/NeurIPS proceedings.

Durable novelty boundary: latent codes, support-conditioned function prediction, test-time adaptation, function representations, generic canonicalization, fixed-basis projection, shared equations with varying coefficients, and neural-to-symbolic extraction are all prior art. It is also false to say that latent variables are unconditionally never identifiable: auxiliary-variable generative assumptions can restore identifiability in a different model class. The defensible candidate is the narrower combination of declared physical probe-response equivalence, exact chart intervention, unseen-entity support-only named equation coordinates, optional rank-aware learned-prior fusion and sealed temporal symbolic evaluation with prediction baselines/tails kept separate. No `first` priority wording is authorized.

The theory document and paper plan now include this boundary. Generic canonicalization is not the contribution; the phrase must be `support-conditioned response-space canonicalization on declared physical probes`. Fixed linear projection is continuous in the probe response, while the independent support-to-response optimizer retains its narrower rank/conditioning and affine-chart guarantees. GIRD remains optional and can become a headline algorithmic contribution only if the frozen real decoder-prior increment passes.

Final SHA-256 values after the literature audit are: paper plan (timestamped and fixed identical) `fc92a691f0a17c2112e3715d68686b675cf95541aae6969b8dbb718cfd7de5e7`; closest-work matrix `556ba794ac99825152a1abc0d9ebd176bd6612233326c7e7d470b1d221f9e3fc`; theory/related-work document `8afda99a6b5a1dcd09d90455c47cf305b519bdbcd3c1fe06048d622bb3158a41`; canonical claim gate `9f454406ada291b8c83aea8f2d4da25ee81d8698aac50afc59d37329d0156a46`. The paper-plan page sum remains exactly 9.00.

At 12:32 CST the authoritative neural/GIRD JSON remained `0 completed / 4 active / 21 pending / 0 failures`, with continuous launcher updates and `confirmation_targets_opened=false`. No partial metric was inspected or promoted.

## 133. Pragmatic expression criterion and paper-branch rule frozen (2026-08-29 12:36 CST)

The user again fixed the mandatory endpoint as a comparatively interpretable and scientifically suggestive compact expression with strict unseen-entity support-to-query physical pooled `R2 >= 0.85`. It need not recover the initial/raw q, a preselected physical variable, a unique law or a microscopic mechanism. Decoder-functional and support/structure-recalibrated coordinates remain eligible. A credible stage-wise clue is operationally evidenced by a predeclared named basis, coefficient stability across seeds or support offsets, and at least one falsifiable domain hypothesis. Predictive superiority and causal truth remain separate stronger claims.

`ICLR_RESULT_TO_NARRATIVE_DECISION_RULE_20260829.md` was frozen before any formal Crystal-Cp cell produced a terminal `result.json`. It prevents post-result story shopping with three branches: GIRD becomes headline only if both the exact 25-cell development gate and single-use temporal incremental-value gate pass; otherwise the paper uses the canonical-response diagnostic branch; a failed Crystal-Cp v4 external expression remains an explicit boundary failure and cannot be relabeled by a neural result. No branch changes the `R2 >= 0.85` expression endpoint or requires original-q recovery.

Decision-rule SHA-256 is `b354d50f75fb467fe3b2abd87cb3f6a8d67f40b98c6efb3dd724ef5830520ec5`. The synchronized timestamped/fixed paper plans have SHA-256 `7b9b5fb74076e8a9943f8b1a9bf383143b24b06233f6006d1d48c2ea59bd5275`.

At this observation the authoritative file was `runs/thermoml_crystal_cp_neural_gird_development_20260829/launcher_status.json` rather than `campaign_status.json`. It reported state `running`, `0 completed / 4 active / 21 pending / 0 failures`, GPUs 2--5 assigned to fold0 seeds 0--3, and `confirmation_targets_opened=false`. The legacy extended campaign remained terminal at 2910/2910 with zero failures. No partial neural metric was inspected or promoted.

## 134. Stage-wise hypotheses audited and Crystal-Cp v4 source corrected (2026-08-29 12:42 CST)

`SYMBOLIC_STAGEWISE_HYPOTHESES_20260829.md` now makes the pragmatic interpretation endpoint independently readable and falsifiable. ZT-H1 tests whether the quadratic coordinate predicts an in-range derivative reversal under denser measurements; VP-H1 tests effective reference vaporization enthalpy and its temperature-correction sign against independent calorimetry; CP-H1 tests whether support-routed heat-capacity upturns persist under measurements extending toward the frozen boundary and are enriched for independent transition evidence. These are stage-wise clues, not causal or unique mechanisms.

The ZT development stability file contains 80 entities under four support offsets. Its pre-existing q-distance Spearman versus offset 0 is `0.978023/0.973300/0.967875`. Direct six-pair recomputation from the 320 coefficient rows gives median/minimum Spearman `0.9017/0.8735`, `0.9739/0.9483`, and `0.9593/0.9397` for the physical intercept, slope and curvature, with curvature-sign agreement at least `0.9625`. Vapor-pressure named-coordinate median offset stability remains `0.9991/0.9762/0.7974`, and joint-coordinate median/minimum is `0.8905/0.8546`. Crystal-Cp v4 selects the inverse-square-root atom in 5/5 folds, degree 2 and delta `0.0003` in 4/5, but routes only `1/3/2/3/3` training entities per fold; it therefore remains a narrow development clue pending confirmation.

The paper plan previously mapped the v4 `R2=0.859300` claim to the older `thermoml_crystal_cp_transition_structure_development_20260829/result.json`, whose actual pooled R2 is `0.839584`. This was a source-path error, not a numerical result change. The authoritative passing v4 source is now correctly mapped to `runs/thermoml_crystal_cp_router_margin_development_20260829/result.json` plus its independent `analysis/decision.json`. The stage-hypothesis audit SHA-256 is `90c69247f461c83c354d7b07bb3d3e56f913c38e11724032cb6eed304a76e4ad`; synchronized paper-plan SHA-256 is `7cef5b7da80469afa832125c360452ca7c1ab95df396bf54659b373e6c9e4497`; updated claim-gate/main-report SHA-256 values are `b16934806f7751349b8f8120d0779754a97b66815be6b95e8d1e23cec6f4d751` and `be7640870455e06696bcfc24455c3bb6fcca55c1cabcd631ac216fbcd2091803`.

At 12:37 CST the formal neural/GIRD launcher remained `0 completed / 4 active / 21 pending / 0 failures`, with zero terminal `result.json` files and `confirmation_targets_opened=false`. The all-development CPU package status remained terminal for seeds 0--2. Local `nvidia-smi` and host tmux access were unavailable in this container, so no conflicting GPU/process inference was made.

## 135. ICLR top-50 completion audit and portability repair (2026-08-29 12:47 CST)

An independent read-only reviewer scored the current requirement coverage as theory `1/2`, unified algorithm `0.5/2`, independent real evidence `1.5/2`, operational interpretability `2/2`, strong-baseline fairness `1.5/2`, negative results/tails `1.5/2`, reproducibility `1/2`, and narrative/page feasibility `1/2`. This `10/16` subtotal is a checklist coverage score, not an acceptance probability. `ICLR_TOP50_COMPLETION_AUDIT_20260829.md` records the authoritative evidence and four hard blockers: terminal learned-method branch decision; all-development/joint-seal/single-use Crystal evidence; an actual 9-page manuscript; and final claim/citation/reproduction/adversarial audits. Its SHA-256 is `e580ec0787f513883b57f439c537af85239a24ea0997e9cd9d84b822c3ca2be0`.

The latest synchronized timestamped/fixed paper-plan SHA-256 after adding this completion audit is `de82fe3947fc5b4c69148f07b943b64ee2b1f20cdad93c20f456147bdbd095c8`.

The repository portability scan found exactly two non-document, non-run hard-coded machine roots: the original gauge benchmark and affine-calibration-extension shell launchers. Both now derive `project_root` from `BASH_SOURCE`, accept `PYTHON_BIN` with portable `python3` default, and fail clearly when torch/numpy/pandas are unavailable. `bash -n`, the environment dependency check and a repeated source-code scan all pass; no `/public/home/wangyg`, `/home/wangyg` or `/tmp/` path remains outside run artifacts, Markdown records or git internals. Launcher SHA-256 values are `c1f2686fc2345edf03fdaf8b10b148ec625e1a31b2e9d0701e0c5bd296704eb6` and `570fa72cd8002f8686649c17aed2689862aa3d823ee47e5bdb732272de3e8f8f`.

The current branch tracks `origin/research/latent-q-stagec-20260826` at `0 ahead / 0 behind`, but the active research worktree remains dirty and new artifacts are not yet committed or remotely synchronized. Treat clean commit/reproduction/remote sync as submission blockers, not as completed merely because an earlier snapshot was pushed.

At 12:45 CST the authoritative neural/GIRD launcher was still actively refreshed at `0 completed / 4 active / 21 pending / 0 failures`, with `confirmation_targets_opened=false`. No partial result was inspected.

## 136. Branch-neutral ICLR Introduction drafted and legacy paper boundary repaired (2026-08-29 12:51 CST)

The `research-paper-writing` workflow was used to draft `paper/sections/introduction.tex`. It has six single-purpose paragraph roles: task, latent-chart challenge, equation-discovery gap, concrete response-canonicalization pipeline, external evidence with adjacent baseline limits, and contributions. `paper/ICLR_INTRODUCTION_AUDIT_20260829.md` contains the required mini-outline, reverse outline, paragraph-level claim--evidence map, self-review and explicit citation gate. The prose survives either Branch A or B: GIRD is described as a rank-aware test and may not enter the title/abstract as a headline algorithm unless the frozen terminal gates pass. Introduction/audit SHA-256 values are `06b296b0bac38d07b4f6dd7afd01a44564d7f124da3e6ed6666eaae6f4586c7e` and `3b20b8d3270ae959ccf0c2dff751fb96ddb105e90f70607f1bacf2e1ffe68c39`.

The old `paper/main.tex` is an AAAI-era historical draft rather than the current manuscript. `paper/README.md` now makes this explicit and points future agents to the ICLR claim/plan/Introduction artifacts. The previously user-excluded dataset was removed from the legacy prose, table and bibliography; a targeted scan of the manuscript, bibliography, current Introduction/audit and paper README finds zero remaining mentions. The legacy draft still compiles successfully to five pages after the removal; it must not be used as evidence for the current submission. README/main/BibTeX/PDF SHA-256 values are `37e816d61856f08416e7603d599f8d5f1632a6700a9aeedc5daef72a03137829`, `433c95d8c8c68d4e2db183947fc6abacb211a54f79d97820c9684ba7a530c586`, `b83f0bcb7f3b8dd896230c2fdf533557277d5c8fbb71cb125ddd5f95aff66d15` and `2769add796370100613017e085efe77989e72010d202be84b0641dd0616633dd`.

The new Introduction citation keys are intentionally marked placeholders until primary-source BibTeX verification; this is a visible blocker rather than a silent bibliography fabrication. At 12:51 CST the neural/GIRD status remained `0 completed / 4 active / 21 pending / 0 failures`, confirmation unopened.

## 137. Current ICLR prose and decision documents made versionable (2026-08-29 12:53 CST)

The historical `.gitignore` ignored the entire `paper/` directory and every root Markdown file except a fixed allowlist. That made the new ICLR Introduction, paper handoff and several frozen decision/audit artifacts invisible to git and therefore unavailable on another machine. The ignore rules now expose only `paper/README.md`, `paper/ICLR_INTRODUCTION_AUDIT_20260829.md`, and `paper/sections/introduction.tex`; legacy paper sources, templates, PDFs and build products remain ignored. The root allowlist now also exposes the fixed/timestamped paper plans, manifest, closest-work matrix, result-to-narrative rule, top-50 audit and stage-hypothesis audit. No experiment output directory was broadly unignored.

`git status --untracked-files=all` confirms exactly those three current paper files are visible, and the root decision documents are also visible. `.gitignore` SHA-256 is `f61d6df7be86444b3aa41d4dd3bbee4a7213e7466729ee045fa703ce7ff84bc4`; latest synchronized paper-plan SHA-256 is `c257ec7940486e2a7f250cd4422b64b7ca469366902642c96a1f0a4987f308b6`. This change makes later commit/push possible but does not itself commit or push the dirty research worktree.

## 138. Common canonical-response and conditional-GIRD Method drafted (2026-08-29 12:57 CST)

The `research-paper-writing` Method workflow produced `paper/sections/method.tex` plus `paper/ICLR_METHOD_AUDIT_20260829.md`. The section defines the unseen-entity support/query setting, probe-response equivalence, the fixed named canonical coordinate, quotient invariance and projection-stability bound, support structure re-q with rank/query-amplification diagnostics, the explicitly affine/full-rank stable-GN calibration scope, nested deterministic GIRD dictionary development, probe-Gram-whitened prior fusion, singular-direction risk decomposition, and the exact rank-aware fallback rule. It is valid under both paper branches: real learned-prior improvement remains a pending claim and cannot enter headline text from this draft alone.

The audit provides the required mini-outline, module design/motivation/advantage table, pipeline sketch, reverse outline, claim--evidence map and self-review. A standalone two-column LaTeX check compiles with no errors, undefined references or overfull boxes and occupies two pages under a generic one-inch-margin article, leaving margin within the planned 2.70 Method pages; the isolated section-title underfull warning is formatting-only. Method/audit/check-PDF SHA-256 values are `cc4ee56cb212bcb52678ffb9cdf53226d4a4f4a8adad09e8084169ab5a5b4040`, `c1c833ff0c27ea4ca0791965deda7f0edbaa24d871031ec4372af2f8df7ffd8b` and `bc250565d9d30ceb40bcf34be9af129e8f7c929edd3792a475a59db46905a05d`.

The `.gitignore` paper allowlist now includes only the current Introduction/Method sources and their audits plus the paper README; legacy templates/build products remain local. Current `.gitignore` and synchronized paper-plan SHA-256 values are `3373021f70fcf0ba9cb404e77ed2ca90c3be85d30ab8bc8c359e4c142f732261` and `db45031eaaeb3be6e3f28926090276aea5868ff41bebd90cb3b64d6c4bb5a986`. At 12:57 CST the formal matrix remained `0 completed / 4 active / 21 pending / 0 failures`, confirmation unopened, with launcher status continuously refreshed.

## 139. Branch-safe ICLR Experiments section drafted (2026-08-29 13:02 CST)

`paper/sections/experiments.tex` now organizes the evidence by four research questions rather than directory chronology. It defines the common unseen-entity/DOI/time protocol and information paths; reports the stable affine gauge intervention and controlled conditional-GIRD result; leads with the one-shot ZT and vapor-pressure expression confirmations; puts coefficient stability, uncertainty, tails and the strongest local baseline adjacent to the headline results; distinguishes raw-q, decoder-functional and support structure re-q evidence; and reports the complete Crystal-Cp development baseline table including the 748 negative predictions and sub-0.85 bootstrap lower bounds. The unopened Crystal section contains only the frozen Branch A/B/C replacement rule and no future metric.

The section uses exactly two booktabs tables and makes no prediction-SOTA, true-q, causal-law or universal-GIRD claim. `paper/ICLR_EXPERIMENTS_AUDIT_20260829.md` contains the required mini-outline, reverse outline, claim--evidence map, fairness/negative-result review and missing-evidence gate. A standalone two-column compilation occupies two pages with no errors, undefined references or overfull boxes. Experiments/audit/check-PDF SHA-256 values are `19a0225eae249bb75f6b1bae2b68299dd876592d612efe8b408ee7397e5584c0`, `e16c2271fe35253081cf72e72770de0e5ecaed6545ab5019f1197d587ce2ea95` and `6c196138ae31d0a297712f6009aa8edc98fb0c6fc142540b3533970dc9cd6a52`.

At 13:01 CST the neural/GIRD launcher remained actively refreshed at `0 completed / 4 active / 21 pending / 0 failures`, confirmation unopened. No partial cell score was inspected or used in the prose.

## 140. Bounded ICLR Related Work drafted (2026-08-29 13:04 CST)

`paper/sections/related_work.tex` compresses the verified closest-work matrix into four mechanism-centered paragraphs: support-conditioned/implicit function representations; latent identifiability, invariant metric structure and canonicalization; learned-coordinate/neural-to-symbolic equation discovery; and the resulting bounded positioning. It explicitly states that context adaptation, fixed basis projection, generic canonicalization, neural-to-symbolic extraction and shared equations with varying coefficients are prior art. The only claimed distinction is the combined audited interface of declared physical probe-response equivalence, support-only unseen-entity coefficients, exact chart intervention, rank/conditioning diagnostics, and sealed temporal expression evaluation. No `first`, generic latent-unidentifiability, unique-basis or universal-prior statement appears.

`paper/ICLR_RELATED_WORK_AUDIT_20260829.md` provides the mini-outline, reverse outline, closest-work distinctions, claim--evidence map and citation-completion gate. The standalone two-column section compiles in one page without errors or overfull boxes; undefined citations are expected because BibTeX records remain intentionally uncreated until primary-source field verification. Related-work/audit/check-PDF SHA-256 values are `aac60cf7ac01cf393386cb760484e1d639b7e039901da97fdc108914aec053f9`, `155344093633cad53465dd69cae127e955320bbb4a1f71855c4dbe49a88bbcca` and `085823bf30fa1602a5f36a75f7640e507f976a778acb1b6cd86c1a80b44452ae`.

At 13:03 CST the formal neural/GIRD matrix remained `0 completed / 4 active / 21 pending / 0 failures`, confirmation unopened. No partial result influenced the Related Work or contribution boundary.

## 141. Interpretability endpoint reconfirmed and branch-safe conclusion compiled (2026-08-29 13:09 CST)

The user reconfirmed that the mandatory expression endpoint is pragmatic: a compact, readable expression passes when strict unseen-entity support-to-query evaluation in physical units reaches pooled `R2 >= 0.85` and its named terms provide a credible stage-wise scientific clue. It need not recover the initially learned raw `q`, a preselected physical variable, a unique true law, or a microscopic mechanism. Decoder-functional, support/structure-recalibrated, and other response-aligned low-dimensional coordinates remain eligible. Predictive superiority over PCHIP, kNN, CNP, FPCA or no-q MLP is a separate stronger endpoint. This is an interpretation of the already frozen threshold and claim hierarchy, not a protocol, formula, branch, or numerical-gate change; the pre-result decision-rule file and its SHA-256 remain unchanged.

`paper/sections/limitations_conclusion.tex` and `paper/ICLR_CONCLUSION_AUDIT_20260829.md` now complete the common Branch-A/B-safe paper prose. The three paragraphs delimit basis/probe relativity and non-causal scope, summarize the confirmed ZT and vapor-pressure equations, and end with the frozen falsifiable ZT turnover, vapor-pressure calorimetry and Crystal-Cp boundary-stage tests. A generic 10-point two-column compile occupies one page with no errors, undefined references, underfull boxes or overfull boxes. Source/audit/check-PDF SHA-256 values are `a8e5a7888b284d1ff618f4e1254ee440e6585ec1b25a11f20dcf1df7f835d9cb`, `671c58203f128c1c9cee261925c2c20964fd287acad7172a5e9e82210339fd97` and `dbc3fa8f16d4ca46849c8f9e00eb26fe4056c68099a2fa0d075f030ca1891ac5`.

At 13:09 CST the authoritative Crystal-Cp neural/GIRD launcher remained actively refreshed at `0 completed / 4 active / 21 pending / 0 failures`, with `confirmation_targets_opened=false`. No partial scientific metric was inspected or used to alter the endpoint or manuscript.

A generic 10-point two-column integration compile of the current Introduction, Related Work, Method, Experiments, and Limitations/Conclusion occupies six pages including a provisional title block, with no LaTeX error, overfull box, or unresolved internal reference after the second pass. Citation warnings remain intentionally unresolved until primary-source BibTeX verification, and the build has no abstract or final figures, so this is a footprint check rather than a submission PDF. Its SHA-256 is `8a340f5f3d34aa20a8dcdf05f4ffb09554a48921d0231e54bb18ccd1db40e1e3`. The synchronized fixed/timestamped paper-plan SHA-256 is `d6636e746802c4db56a1a7587e00501dbbbbf6f3de8c5103ba4ec8378ef1e2a9`.

## 142. First neural/GIRD cells finish; branch-neutral hero figure and citation audit begin (2026-08-29 13:24 CST)

At 13:20 CST the authoritative neural/GIRD launcher advanced to `4 completed / 4 active / 17 pending / 0 failures`; the four fold-zero seeds 0--3 were recorded complete and the controller dispatched fold-zero seed 4 plus fold-one seeds 0--2. `confirmation_targets_opened=false`. No completed-cell metric or partial aggregate was inspected.

The `academic-figures` workflow produced two deterministic, pure-vector Figure 1 drafts. Variant A is a `183 x 76 mm` horizontal four-panel flow and is preferred for the nine-page paper; Variant B is a `183 x 154 mm` 2-by-2 alternative. Both show only branch-stable content: raw-q gauge ambiguity, the fixed-probe response quotient, named response coordinates with a solid support-re-q core and dashed optional rank-aware prior, and the already sealed ZT/vapor-pressure temporal expression results with the interpolation caveat. XML parsing, unique IDs, editable text attributes, Python compilation and `git diff --check` pass. Generator/Variant-A/Variant-B SHA-256 values are `4b311494c8d77717fbca93aaacbb3fd3304deb39387651346cf900c003203c59`, `5219ceeebbbaea3667c2784fd378ea08db23b4502ae4a4b62f0d00f0d1c6244a`, and `1952acb075c871d11faee819a9fa1d9e3079c9f7780a152b5a04dcf88ab6fc0f`. The local symbolic-icon rasterizer does not preserve the intended color palette, so its PNGs are QA previews only; SVG is authoritative.

The `citation-audit` workflow is now running one fresh official-source web review per cited key. The first six keys are real and relevant; two citation sentences were narrowed after reviewers rated their original scope WEAK: the Introduction no longer generalizes Locatello's result to arbitrary regularization, and Related Work separates Functa's shared-prior training from partial-observation MAP inference. `paper/iclr_refs.bib` contains only the six official records already returned, while the remaining keys are still pending. Current Introduction/Related-Work/bibliography/context-ledger SHA-256 values are `8930d34bbe6cfc944f16c0a93403bc2c68f0e824ec22cf386824aebf9e346a04`, `b0e21b63671342ad00d1ffb77b7caf14dd89a89a1cec507bf5a70910745f7600`, `493b537df713d47481be47d3b2e1ce846a57ea561f79b99cc15ceb8f587eaf77`, and `2cc88872fcacb694c963b6b538635ea729566d04e2185883b7599a5d9684e8fc`.

## 143. Current citation audit passes and Figure 2 is independently corrected (2026-08-29 13:49 CST)

The current ICLR draft now has a terminal same-family provisional citation audit. All 15 cited keys were assigned to fresh `gpt-5.6-sol` `xhigh` reviewers with official-source web checks; every work exists, current metadata is verified, and every current use is supported after scope-tightening rewrites. The rewrites separate Functa prior learning/adaptation/MAP completion, restrict the Locatello and Khemakhem identifiability claims, describe UPINN as recovering neural representations of unknown differential-equation terms, scope parametric equations to parameterized instances, and state LaSR's LLM-induced abstract concept library concretely. `paper/CITATION_AUDIT.json` records `PASS`, `15 KEEP / 0 FIX / 0 REPLACE / 0 REMOVE`; its current SHA-256 is `4fea76450857c77a99c117af83a5f5441dc6420a665e1ca051b977c9ef89caf0`. The human report SHA-256 is `4d64c4d25ef2270e202a4250bb74af924b5b6dac196d45ecaa982b2a37396e27`, and the exact current contexts hash is `cb87e3c57862114ae295404182d7d6c2f71c40628e84b1c5dc66af9bac6ea454`. Additional dataset-source or numerical-method citations introduced later must be separately audited; this PASS applies to the 15 citations presently in the draft.

`paper/iclr_draft.tex` is now the active generic two-column integration wrapper, distinct from the legacy AAAI `main.tex`. A full LaTeX → BibTeX → two-LaTeX-pass build produces eight pages with no undefined citation/reference, LaTeX error, BibTeX warning, fatal error, or overfull box. Wrapper/PDF SHA-256 values are `caf7b47deaefb6fd282c574f2a7e0b3b7e1b90aaf1a39f3eb8d45049516526a9` and `f968c5ed99db71e010bffb3cbb7b998d22aea0ed31b652803c8f0501864b4500`. This remains a footprint check rather than final ICLR styling and does not yet include the abstract or final figure/table placements.

Figure 2 was regenerated directly from the sealed stable-gauge and controlled-GIRD CSVs and then reviewed by a fresh independent figure auditor. The auditor confirmed every numeric value but rejected three presentation choices: `75 independent interventions` was unsupported because the design is 3 families × 5 seeds × 5 gauges; the dashed denominator was ambiguously called direct support even though it is support-only GIRD-GN at `lambda=0`; and the bars lacked distributional uncertainty and selection-protocol detail. The final artifact now says `75 family--seed--gauge cases`, labels the denominator exactly, defines bars as ratios of medians with lower better, states the fixed entity-ID-modulo-5 outer-training validation and 1%-of-best selection rule, explains why direct-target OMP remains in the main table rather than the denominator, and adds deterministic paired entity-bootstrap 95% intervals with 10,000 resamples. The panel-b label/legend collision is fixed. PDF/value-ledger/generator/caption SHA-256 values are `4fe94201a330845e858e2d22c586867e91b304381cdc0285251980f5c5dcb477`, `8d7686b56fb169ef63cfbc5bda8fad3e1228cd6e89114509428b76ad6a371eac`, `e0fbfc1dc3b348ee2fc403a4a048c83ccfb3b63611f0a9f1d6e0b641c0e8e398`, and `07ecfbdf8f35733fc7415e38f71ac1e2dfbae5ca83cbf450833424bf7fb85f88`. Two consecutive regenerations are byte-identical; the standalone one-page LaTeX include has no errors or overfull boxes; all PDF fonts are embedded CID TrueType with no Type 3 fonts.

`paper/README.md`, `MANIFEST.md`, `.gitignore`, `PAPER_PLAN.md`, and the synchronized timestamped plan now expose and describe the current draft, citation ledger/traces, and both paper figures. The two paper-plan files remain byte-identical at SHA-256 `898d147170330d9173ffd4e301d25e577975dd57d78950f1deb10e770826a632`. The citation HTML reader view was re-rendered after its first audit found collapsed metadata lines; the metadata is now a proper table and a fresh render-fidelity reviewer returns PASS with zero blockers/warnings. HTML/review-sidecar SHA-256 values are `0cc45bfdcb78074451030a4bf50a7cb1c8a9be5aae83e581cc0ff44764ef8966` and `ba12613fc1f2f6ca2cccb4d98b581fb3f7f35b641035e9e8a65916e337f72459`. The MD/JSON remain authoritative.

At 13:48 CST the authoritative Crystal-Cp neural/GIRD launcher remained `4 completed / 4 active / 17 pending / 0 failures`; active cells were fold0 seed4 and fold1 seeds0--2 on GPUs 5,4,3,2 respectively, `confirmation_targets_opened=false`, and `maximum_used_mib=1024`. No partial cell metric was inspected or used. The unchanged count over this documentation interval is an observation, not a failure or hang claim.

## 144. Figure 3 passes independent scientific and geometry audit (2026-08-29 14:13 CST)

`paper/figures/figure3_real_transfer.pdf` is now the complete four-panel real-transfer figure. Panels a/b use a deterministic post-scoring representative rule—entity nearest the expression-family median entity NRMSE, with lexicographic tie breaking—rather than a favorable hand-picked example. The selected entities are Sc2Te3 (ZT, error rank 15/30, 9 support and 24 query rows) and N-methylethylenediamine (vapor pressure, rank 42/84, 6 support and 16 query rows). The ZT reconstruction exactly reproduces all 919 sealed query temperature/target pairs across all 30 entities under the frozen stable-temperature-sort/every-fourth-row support rule.

The paired entity panel reports the exact complete comparisons: the expression beats kNN on 16/30 ZT entities and PCHIP on 37/84 vapor-pressure entities, with no missing rows or ties. The development-stability panel reports median/minimum cross-offset coefficient correlations: ZT level `0.901735/0.873511`, slope `0.973922/0.948289`, curvature `0.959283/0.939662`; vapor-pressure level `0.999050/0.998573`, enthalpy `0.976188/0.973208`, and heat-capacity term `0.797364/0.697435`. The caption explicitly separates development coefficient stability from temporal confirmation and does not claim predictive superiority.

A fresh independent reviewer returned PASS with zero hard failures and zero warnings after the a/b/c legends were moved outside the axes. Legend/data, legend/axes, legend/title, and panel-c annotation/data intersections are all zero. The PDF is pure vector (`pdfimages=0`), uses embedded CID TrueType fonts with no Type 3 fonts, and has color/shape/line-style redundancy. The deterministic generator is `scripts/generate_iclr_figure3_20260829.py`; the current source ledger is `paper/figures/figure3_values.json`. Current generator/PDF/PNG/JSON SHA-256 values are `83ff6bf9f9d3e8c783203fa1ec751d463b1ac396429c68549a60caebe90d12df`, `3bae589f3df93a2479d97ca7c5ba5f883e1f3592f993ea3f333cf6db3eb67467`, `87315cc735e239f394166bfa89337a030a110bbca152b7d8bbc448fba2487769`, and `7708d013258caba205dc58b668c2b687b304f9c009c3899d91f59e81a88be1b7`. Regeneration is byte-identical within the pinned project environment; a fresh incremental review confirmed that the earlier Matplotlib 3.11.0 versus current 3.10.9 byte difference is only renderer/font-subsetting variation with identical numeric payload and geometry. Cross-environment PDF bytes are not treated as scientific evidence, so the ledger and source hashes remain the primary numerical provenance.

## 145. New `/tmp` cache payload migrated; persistent-storage rule tightened (2026-08-29 14:13 CST)

A new top-level `/tmp` audit found no experiment output, dataset, model, paper source, figure, log, or result attributable to this project. Two inactive project-created runtime caches were recoverably moved to `runs/_runtime_cache/legacy_tmp_migration_20260829_1415/`: a 111,804-byte Matplotlib font list and an 11,524-byte TeX bitmap font. Their source directories `/tmp/latent-mpl` and `/tmp/texfonts` are absent after the move; `MIGRATION_MANIFEST.md` records exact destinations, sizes, and SHA-256 values.

`/tmp/torchinductor_wangyg` remains temporarily in place because the Crystal-Cp neural/GIRD campaign is active. It is a regenerable 45 MiB runtime cache, not research material; moving it during active CUDA work could disturb another live process. Archive or discard it only after the campaign is terminal and ownership is rechecked. System IPC, tmux sockets, Codex sandbox mounts, and unrelated paper/project files were not touched.

The durable rule is now explicit in `README.md`: every new project command must place outputs, fixtures, logs, downloads, Matplotlib/XDG/TorchInductor caches, and TeX variable/config/cache trees below the public repository, normally `runs/_runtime_cache/`. Do not use `/tmp` for persistent project state.

## 146. Abstract and audited Figures 2--3 integrated into the current wrapper (2026-08-29 14:20 CST)

`paper/iclr_draft.tex` now includes `sections/abstract.tex`, `figures/latex_includes.tex` (Figure 2), and `figures/figure3_latex_include.tex` (Figure 3), with `graphicx` and an explicit two-column top-float fraction. The float setting is necessary because the full-width Figure 3 otherwise deferred to a page after the bibliography; the current build places Figure 2 on page 5, Figure 3 on page 6, the main expression tables on page 7, and Limitations/Conclusion before the references begin on page 8.

The generic two-column BibTeX build is nine pages total, has no undefined citation/reference, LaTeX error, BibTeX warning, fatal error, or overfull box, and has wrapper/PDF SHA-256 values `b90905072a98392e3d0250449737ce00d68989cbde8ebbb058e0885d7a9aaae4` and `a2249fb7a1e49984d2d3d95e42a94b30cc8a8eef3cad6532ff88c9815136b8a8`. This is still a generic footprint build, not the official ICLR template; Figure 1 remains SVG-only and is not yet integrated.

The wrapper edit introduced no new citation and changed no audited context. The exact cite set remains 15, the Introduction/Related Work/context-ledger hashes remain unchanged, and `paper/CITATION_AUDIT.{md,json}` now records the wrapper-only hash refresh explicitly rather than pretending a new semantic citation review occurred. The derived HTML was re-rendered from the refreshed MD/JSON and a fresh same-family provisional reviewer returned PASS with zero blockers and zero warnings. The current MD/JSON/HTML source hashes are `79efc930410987cfd864935729eef37de4e73f7f736386acd7eb5d1e507687de`, `dccc1a1de4eb531d6e1d791843556e151efe5f5b1923ce575fd530b64ac182fb`, and `9c1f1f8304a8fec74d377101840ef0c30befe651df9c56651bb8c6b3c2f8e5a7`; the render trace is `paper/.aris/traces/render-html/2026-08-29_run03/`.

At 14:25 CST the authoritative Crystal-Cp launcher remained actively refreshed at `4 completed / 4 active / 17 pending / 0 failures`; active cells were fold0 seed4 and fold1 seeds0--2 on GPUs 5,4,3,2, `confirmation_targets_opened=false`, and `maximum_used_mib=1024`. No partial metric was inspected.

## 147. Pragmatic expression endpoint clarified; Figure 1 passes independent audit (2026-08-29 14:44 CST)

The user clarified that the paper's expression objective is deliberately
pragmatic: the final compact expression is comparatively interpretable when
strict unseen-entity support-to-query physical pooled `R2 >= 0.85` and its terms
offer some scientifically useful stage-wise insight.  It does not have to reuse
or recover the first learned q, a preselected physical variable, a unique law or
a microscopic mechanism.  This is not a numerical threshold, formula or branch
change.  `PAPER_PLAN.md`, its timestamped mirror, the canonical claim gate, the
stage-hypothesis audit, the beginner-readable report and the Introduction now
make the core endpoint and the stronger corroborating evidence distinct:
coefficient/support-offset stability and falsifiable follow-up hypotheses
strengthen paper credibility but do not make a particular raw-q chart the
target.  The two paper plans are byte-identical at SHA-256
`dfe5113ae4dc8d973c228fce9a0ee40febb6969f092197deadaa874e4ddff114`.
The synchronized canonical claim gate, stage-hypothesis audit, and main report
SHA-256 values are `28e25e8291f90b0349ff3d0aa517f8f014862b492f1deaf35c4c1ddc17655e08`,
`4df4821275fff353481911a6681b183a1d6e6b96afe073f5ea7d9479b8e65b5d`,
and `dc801c793abea8b2371c3d51afcdc361cc0f27e40d2c4cf1214fd098eabaca74`.

Figure 1 is now integrated as a paper-ready pure-vector PDF.  A fresh independent
review initially found Panel-a text overflow and Panel-d curve/label collisions;
the generator was minimally corrected by unifying the SVG/PDF DejaVu font stack,
shortening overflowing labels, separating chart boxes, increasing minimum text,
darkening small orange/green text, making the baseline caveat domain-neutral,
and moving the evidence curves away from their labels.  The final incremental
review returns PASS with zero hard failures: all text is contained, no connector
crosses an object, the ZT label has about 0.765 mm clearance, the PDF has no
raster images or Type 3 fonts, and all fonts are embedded CID TrueType.  Two
same-environment regenerations are byte-identical.  Generator/SVG-A/SVG-B/PDF
SHA-256 values are `9f8574a4b340416b42405cb61a3b3a86f0b868293f048b6c975311ff27d91109`,
`0659b19b9fdcd82dc76bce175e892e7bcca509f68a695acc51b862ff3295f963`,
`5b8e08a2a2810a38a66ece9def1e5fdf48f50520366f4b4e29bc61cb385811f2`
and `a6983f5a0f5e0dfd32bb831866e693806c39aea755ba377c342ab8417311f72f`.
`paper/figures/figure1_values.json` binds the two displayed sealed R2 values to
their exact decision files and records the non-SOTA/non-causal claim boundary.

The current generic two-column paper compiles to ten pages, with main text
through the conclusion ending on page 8 and references on pages 9--10.  There is
no undefined citation/reference, LaTeX error, BibTeX warning, fatal error or
overfull box.  Wrapper/Introduction/PDF SHA-256 values are
`9fd243893391a43ff01d42dc867ad0ad23a83eac047075717ae2be409823383b`,
`f788329d95701428e9c663da1490bbf0a198ab85dba9d0d60ee9b7773ed6a68b`
and `3a70fa07b2cd655193726f73aa31b5190a2f9ca6d4fd5202c39e7645ca972420`.
The citation ledger records that the Figure-1 and non-citation endpoint wording
refresh changed no cite key or audited citation context; its HTML reader was
regenerated and a fresh same-family provisional render-only reviewer returned
PASS with zero blockers and zero warnings.  HTML/review-sidecar SHA-256 values
are `dcd5f287c18f3d22dd993a88bcf0577455b3dd791bcb2db2db07abaf204d16c2`
and `a90e69abb7be9180417af505bea5316aa1a1dad51d7d774aaa9fb1c1787f77c4`;
the trace is `paper/.aris/traces/render-html/2026-08-29_run04/`.

One generic LaTeX compile exposed that the system default `VARTEXFONTS` still
pointed to `/tmp/texfonts`.  The newly created 11,524-byte font cache was moved
immediately to `runs/_runtime_cache/texfonts/`, the `/tmp/texfonts` directory is
absent, and every subsequent compile explicitly binds `TEXMFVAR`,
`TEXMFCONFIG`, and `VARTEXFONTS` under the repository.  No research artifact was
left in `/tmp`.

At 14:43 CST the Crystal-Cp neural/GIRD launcher remained actively refreshed at
`4 completed / 4 active / 17 pending / 0 failures`, with fold0 seed4 and fold1
seeds0--2 active, `confirmation_targets_opened=false`, and no partial scientific
metric inspected.

## 148. Dataset/numerical provenance and all 21 current citations close cleanly (2026-08-29 15:15 CST)

The pragmatic expression endpoint from Section 147 remains the controlling
criterion: a compact final response-aligned expression is successful at strict
unseen-entity support-to-query physical pooled `R2 >= 0.85` when its terms offer
scientifically useful stage-wise insight. It need not recover the original
latent `q`, a preselected physical coordinate, a unique true law, or a
microscopic mechanism. Expression fidelity, scientific interpretation, and
predictive comparison against strong nonparametric baselines remain separate
claims.

The draft now cites and freezes the missing dataset and numerical-method
provenance. Official-source checks cover the Starrydata project/dataset paper,
the NIST ThermoML archive and IUPAC schema, the PCHIP algorithm, functional data
analysis/FPCA, and the NIST Chemistry WebBook Shomate form. The WebBook entry was
corrected from an invalid publication-year interpretation to an edited NIST
resource with `n.d.` publication year, a separate `Data last updated 2025`
note, access date, editors, publisher, and DOI; a fresh post-fix reviewer marked
it KEEP. `paper/DATA_PROVENANCE.md` binds the local Starrydata and ThermoML
artifacts by exact path, byte count and SHA-256 rather than relying on mutable
upstream `latest` links. Its SHA-256 is
`b9b03f086e8f221906f4703d31dba5ee682c2a46c3c1dcae9e42199e119bea92`.
The bibliography and Experiments-section SHA-256 values are
`ef1e91afe1366e4f2b78ce7f716816d120bf2ba9812423f0feef1bd4fe8b5df1`
and `f1281cb876e905c0c8937310e701665304fb8a5bb601e9e6d20a147e5b7c7323`.

The terminal citation ledger is now `21 KEEP / 0 FIX / 0 REPLACE / 0 REMOVE`.
Its canonical MD/JSON SHA-256 values are
`d695bab18be8655f05274b6ddcacaa948bcea29774bb2d270d11e5bd9f1691ba`
and `10c6cab4bcee88e76b9474bd7a0957199bf75db75dccae983abe340efa90ca8a`.
The reader HTML was re-rendered with the complete JSON sidecar and a fresh
same-family provisional render-only reviewer returned PASS with zero blockers
and zero warnings. HTML/review SHA-256 values are
`32a8b284f2372cf50d3e5d63bf42b22182f1e450bdf07211ab8cf222b7cdfa20`
and `0bdfed834114d1c1276b941a5b7315e9b174e0c689f05da0cef109999f525c17`;
the trace is `paper/.aris/traces/render-html/2026-08-29_run05/`.
`paper/DATA_PROVENANCE.md` is explicitly unignored for remote handoff.

The generic two-column draft still compiles to ten pages: the conclusion ends
on page 8, references begin later on page 8 and continue through page 10, with
no undefined citation/reference, LaTeX error, BibTeX warning, fatal error or
overfull box. Its current PDF SHA-256 is
`3af6eacbc66e38177a0dffc69521c4dcfd659d3029528310c4e891b611a9e272`.
At 15:12 CST the authoritative Crystal-Cp neural/GIRD launcher was actively
refreshed at `4 completed / 4 active / 17 pending / 0 failures` on GPUs 2--5,
with `confirmation_targets_opened=false` and `maximum_used_mib=1024`. No
partial-cell scientific metric was inspected or promoted.

## 149. Theory/evidence scopes separated and official ICLR 2027 build closes the page-format blocker (2026-08-29 16:33 CST)

The pragmatic expression endpoint remains unchanged: strict unseen-entity
support-to-query physical pooled `R2 >= 0.85`, finite target-blind predictions,
and a compact expression whose named terms offer a stage-wise scientific clue.
It does not require recovery of the first/raw `q`, a unique law, or a
microscopic mechanism; predictive superiority remains a separate claim.

A fresh theory--experiment alignment audit found that the manuscript had
compressed two different gauge results into one paragraph. The corrected Method
and Experiments now separate: (1) 375 exact affine decoder counterfactuals
(3 families x 5 seeds x 25 gauges), where raw coordinates change by as much as
`4.9958` while prediction/fixed-basis coefficient changes are only
`4.50e-15/2.04e-14`; and (2) 75 independent stable-GN recalibrations
(3 families x 5 seeds x 5 gauges), where QR/SVD plus a response-loss margin
reduces the old normal-equation response discrepancy `0.0448354` to
`3.6567e-10` and coefficient discrepancy to `6.2327e-9`. The former supports
the algebraic response-equivalence quotient; the latter supports numerical
affine equivariance only under paired centroids, synchronized latent injection,
full-rank support Jacobians, fixed response-space line search and exact-theory/
stable-finite-precision qualifications. `paper/sections/method.tex` also makes
the quotient claim an explicit proposition, adds the query-amplification bound,
states GIRD's independent zero-mean directional-risk assumptions and null-space
limitation, and scopes the complete method registry to the Crystal spread
regime. `paper/sections/experiments.tex` now uses `c_i` for canonical expression
coefficients, never raw neural `q_i`; lowers the controlled GIRD wording from
universal validation to consistency with the bias--variance mechanism; and
states the ideal-gas/negligible-liquid-volume/finite-range limitations behind
the effective vapor-pressure coordinates. Method/Experiments SHA-256 values are
`10380c5787a8d3da4c092963e5f8efac608034ca75b3a86c9709941f17e1f310`
and `56b21526c644555702c5a279fdf3e1f2720c01fba90d502fbe1c56f69cf56091`.

The official ICLR 2027 Author Guidelines are now published and require no more
than nine initial-submission main-text pages; references, appendices, the
mandatory AI-use statement, and recommended reproducibility statement do not
count. The unmodified official style ZIP was downloaded from
`https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip` with
SHA-256 `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`.
Its four required style assets are tracked under `paper/iclr2027/`.
`paper/iclr2027_draft.tex` plus
`paper/sections/ai_reproducibility_statements.tex` compiles under the official
anonymous style to 12 US-Letter pages: main text ends exactly on page 9, while
AI use, reproducibility and references begin on page 10. The settled build has
no undefined citations/references, overfull boxes, BibTeX warnings or label
drift. `scripts/build_iclr2027_paper.sh` resolves the repository root from its
own location, keeps all TeX caches below `runs/_runtime_cache/`, fixes
`SOURCE_DATE_EPOCH`, and produces byte-identical consecutive PDFs. Wrapper,
statement, build-script and final-PDF SHA-256 values are
`1148f851857659b7dd418cdcef98bfbd18bec9125f45cf3330d601b17dfcbf95`,
`c9a39023b6be4cb5b48f884abd887c7e1a1e988944d6931a1220ab3520976f0c`,
`0b854637b5e08b35688a4d2d7a662b96812d125c36be436651aec2d5a66474de`,
and `bef9f70a1c1ed2677aa6eff04306b2a6513e00ddf332c612600382e8125c4322`.

The 21-entry citation audit remains PASS after the non-citation theory/notation
refresh. The current reader HTML has a fresh same-family provisional render-
fidelity PASS with zero blockers/warnings; review sidecar SHA-256 is
`4159bae450232de0a280abd3b84439707c474e5689a20e4cd97bbc50ba1fd146`
and the trace is `paper/.aris/traces/render-html/2026-08-29_run07/`.
The last machine-local executable path in
`scripts/launch_no_id_symbolic_real_datasets.py` was removed in favor of
`PYSR_PYTHON`/`LVS_PYTHON` or the active interpreter; frozen receipt-bound
launchers retain only repository-relative `.venv-lvs-gpu` paths, documented as
such.

At 16:33 CST the authoritative Crystal-Cp neural/GIRD launcher was actively
refreshed at `8 completed / 4 active / 13 pending / 0 failures`, with
`confirmation_targets_opened=false` and `maximum_used_mib=1024`. No active-cell
scientific metric was inspected. The top-50 audit now scores `11.25/16`: the
page-format blocker is closed, but the terminal 25-cell aggregate, one-shot
Crystal chain, final claim-to-raw/clean-clone audit, reviewed commit/push and
post-terminal adversarial review remain open.

## 150. Zero-context paper claims are numerically clean; evidence-completeness audit is being closed (2026-08-29 17:18 CST)

The pragmatic expression criterion and paper narrative are unchanged. A fresh
GPT-5.6-Sol-ultra zero-context paper-to-raw audit first found one real reporting
error: the controlled-GIRD relaxation endpoint had been printed as `0.03436`,
where standard five-decimal rounding of the raw
`0.03435461077597177` is `0.03435`. This was corrected. The same audit also
exposed several over-broad provenance/configuration sentences; the manuscript
now says `float64 direct least squares`, scopes the lambda-zero observation to
four-support, attributes the `0.0448` discrepancy to the original GN extension,
narrows frozen provenance to the actual hashed cohort manifests, and removes a
false claim that direct-target OMP appeared in the main table. The already
frozen Crystal confirmation Markdown contract is now mirrored by
`THERMOML_CRYSTAL_CP_SINGLE_USE_CONFIRMATION_CONTRACT_20260829.json`, which
serializes the cohort, one-shot transaction, expression endpoint, package,
GIRD-promotion and failure-reporting rules without changing the protocol or
opening a target.

A second fresh full audit found `0` number mismatches, `0` config mismatches,
`0` aggregation mismatches, `0` scope overclaims and `0` unsupported claims
across 50 deduplicated claim groups (`21` exact, `25` standard-rounding). It
remained `FAIL` solely because four experimental-design clauses were absent
from that audit's compact evidence boundary. The underlying structured results
already contain three of them: all 15 exact-counterfactual cells record
`gauge_count=25`; all 15 stable-calibration manifests record
`stable_solver=float64_lstsq`, `gn_steps=15` and the response-loss acceptance
tolerance; and exactly two of the 30 GIRD regime results have a failed selected
OMP certificate. The GIRD analyzer was minimally extended to emit
`analysis/protocol_evidence.json`, explicitly recording the modulo-5 nested
folds, 1% smallest-finite lambda rule, identical FPCA folds, `30` certificate
cells and the two failed thermodynamic-seed-1 regime IDs. Re-analysis retained
the old decision, method-summary, entity-metric and audit CSV hashes byte for
byte; only the new evidence file and regenerated analysis manifest are new.
The targeted analyzer tests pass `5/5`. A third fresh full audit with the
complete per-cell evidence is active; until it returns, no canonical PASS is
claimed. Run-01 and run-02 failures are retained as forensic traces rather than
discarded.

The revised official ICLR build remains 12 US-Letter pages with main text
ending on page 9, produces the byte-identical SHA-256
`2b7d053741b3eccb20c7846abe3f3058918e8983d778865b25b9bb9ab09d920d`,
and has no undefined reference/citation, overfull box, LaTeX/BibTeX error or
label-rerun warning. At 17:17 CST the authoritative Crystal launcher was still
actively refreshed at `8 completed / 4 active / 13 pending / 0 failures` on
GPUs 2--5, with `confirmation_targets_opened=false`; no active-cell metric was
read.

## 151. Run05 disclosure audit repaired; official appendix and Figure 3 close QA (2026-08-29 20:51 CST)

The pragmatic expression endpoint from Sections 147--150 is unchanged. A
fresh zero-context GPT-5.6-Sol-ultra paper-claim audit (Run05) found no numeric,
aggregation-value or unsupported-result contradiction, but returned `FAIL`
because the manuscript did not disclose several exact protocol conventions.
Its deduplicated ledger contained 53 claim groups: `23 exact`, `16 rounding`,
`7 ambiguous`, `5 missing`, `1 config mismatch`, and `1 scope overclaim`.
The missing or ambiguous material was the controlled-GIRD regime-level lambda
selection and five-seed pointwise aggregation; the unit of the two failed OMP
certificate audits; NRMSE, ZT temperature scaling, vapor reference temperature,
and scaled-condition-number definitions; log-pressure PCHIP; real-domain seed
aggregation and probes; Crystal normalized coordinate and router configuration;
OOF versus unopened-final Crystal hyperparameters; support/rank conventions;
and over-broad byte-count and basis-residual wording. Run05 is honestly archived
at `paper/.aris/traces/paper-claim-audit/2026-08-29_run05/`: compaction made the
verbatim response unavailable, so the trace identifies itself as a summary
rather than fabricating a transcript.

The manuscript now states each convention directly. In particular, Figure 2's
tested GIRD applies the validation-selected lambda uniformly within each
predeclared family--seed--support regime; rank is a diagnostic and no per-entity
rank router generated the reported values. The vapor baseline is named
log-pressure PCHIP, all reported neural medians identify their seed and
pointwise aggregation, the exact ZT scaler is
`mu=280.58430016384204 K`, `sigma=193.38620745002135 K`, the vapor reference is
`T_r=362.72 K`, and the Crystal appendix separates the five OOF selections from
the unopened final package. `paper/sections/protocol_details.tex` is the frozen
domain-protocol appendix (SHA-256
`a90cf35189e38ffa41530e843dfbd1b545b0a2e16dd830d5c9dba47ae3c7092f`).

The official ICLR PDF is now 13 US-Letter pages, with the counted main text
still ending on page 9. The nonfloating Appendix Table 3 title and table body
are kept together on page 13; rendered pages 12--13 were visually inspected.
There is no undefined citation/reference, overfull box, LaTeX/BibTeX error,
fatal error or rerun warning. The PDF SHA-256 is
`383945803b692a0c5be1f9316380ee77cd0222e6df39ea4b3b49ae96392d1482`.

Figure 3's metric space is now explicit in both labels and prose. A fixed SVG
hash salt makes two same-environment regenerations byte-identical. The Nature
figure static preflight returns `12 PASS / 0 FAIL / 2 WARN`; the warnings are
only the absent TIFF and a 300-dpi preview PNG, while ICLR consumes the editable
vector PDF/SVG. SVG XML parsing passes, the PDF uses embedded TrueType fonts
and no Type 3 fonts, and visual inspection finds no overlap. Generator,
SVG/PDF/PNG/values SHA-256 values are
`a165dc76f52cbce66f608fd4c9c1d7978d1b1739eeb2a243ad9e02a027003d5a`,
`a19a5146cd8ea024e21b44d508ea2cbb844122d7e5c1f942b5794d07876d5595`,
`f398046ddc013ec200cb7e4c8bd4822c12a6f1e68c8bf8650eeb0a2b7ecadafd`,
`6436ee98c41a9d8427fe5bf5cd3b5a4655d2e0d38b12c02226a22148bb1e1ec1`,
and `6302aade3099385a782f0819fbe4d4e8e9c471b17ae24dba32dbd2906909e6c0`.
The targeted GIRD test file passes `9/9` in the pinned GPU environment. A
separate general-environment `uv sync --extra dev` was stopped after a long,
output-free dependency download; it neither touched the pinned GPU environment
nor any running experiment.

At 20:51 CST the authoritative live launcher reports
`13 completed / 3 active / 9 pending / 0 failures`; active cells are fold2 seed4
on GPU2, fold3 seed0 on GPU3, and fold2 seed3 on GPU4. Fold2 seed2 completed
since the previous snapshot. `confirmation_targets_opened=false`, and no
active-cell scientific metric was inspected. This sandbox cannot access the
host tmux sockets or NVIDIA driver (`Operation not permitted` / NVML failure),
so the refreshed live JSON remains the authoritative status source. A new
zero-context Run06 claim audit is active against the repaired manuscript and a
larger raw-evidence boundary; no canonical PASS is claimed until it finishes.

At 21:00 CST fold2 seed3 reached a terminal success, so the refreshed launcher
advanced to `14 completed / 2 active / 9 pending / 0 failures`. Fold2 seed4 and
fold3 seed0 remain active on GPUs 2 and 3; the guard has not yet assigned a new
cell to the released slot. `confirmation_targets_opened=false`, and no active
or newly completed scientific metric was inspected during this status refresh.

The broader claim-path unit suite then passed `46/46` in 41.63 seconds. It
covered exact and independently recalibrated gauge experiments, their analyzers,
controlled GIRD and protocol evidence, the frozen Crystal neural/GIRD decision
preparer, Crystal transition-structure analysis, and the single-use confirmation
contract. Pytest's base directory was explicitly placed under `public/qa/`.

At 21:12 CST fold3 seed0 also reached a terminal success. The authoritative
launcher therefore reports `15 completed / 1 active / 9 pending / 0 failures`;
fold2 seed4 remains active on GPU2, and the availability guard is waiting rather
than claiming GPUs that are no longer free. Confirmation remains unopened, and
this refresh did not inspect any cell metric.

Run06 subsequently identified one confirmed manuscript protocol blocker while
the rest of its ledger is still being completed: the frozen vapor neural bridge
uses `SEEDS=(0,1,2)` and aggregates three physical-pressure predictions, whereas
`paper/sections/experiments.tex` and `paper/sections/protocol_details.tex`
currently say five seeds. This does not change a reported numerical result, but
it is a real aggregation-description error. Do not edit the active Run06 input
set; after its complete blocker list is archived, change both statements from
five to three and require a fresh zero-context Run07 audit. Also keep the early
`thermoml_crystal_cp_transition_structure_development_20260829` configurations
separate from the current v4 `thermoml_crystal_cp_router_margin_development_20260829`
OOF/final configuration evidence.

At 21:27 CST fold2 seed4 completed successfully. The matrix is now
`16 completed / 0 active / 9 pending / 0 failures`; the guarded launcher remains
in `running` state and is waiting for an eligible GPU rather than terminating or
claiming an occupied card. Confirmation remains unopened, and no cell metric
was inspected.

## 152. Run06 isolates six disclosure errors; all are minimally repaired (2026-08-29 21:40 CST)

The complete fresh Run06 review returned provisional `FAIL` over 67 deduplicated
claims: `42 exact`, `19 rounding`, `1 ambiguous`, `3 config mismatch`,
`1 aggregation mismatch`, and `1 scope overclaim`. It found no headline-number
contradiction. The six remaining issues were all protocol or display-scope
wording: a stale claim that GIRD fused only under a rank rule; a blanket direct
`lstsq` statement that did not describe the Crystal boundary SVD fitter; failure
to distinguish global headline ZT scaling from outer-fold neural scaling; the
incorrect five-versus-three vapor neural seed count; failure to distinguish the
fixed temporal vapor reference from development fold references; and calling a
displayed cross-chart response discrepancy a displayed projection residual.

All six are now minimally repaired without changing any result. Experiments now
state that GIRD's validation-selected weight is fixed within the predeclared
support regime and rank/conditioning only diagnose entities. Appendix A states
that headline ZT/vapor fits use unweighted float64
`numpy.linalg.lstsq(rcond=None)`, while the Crystal boundary fit uses
support-column scaling and explicit float64 SVD. The ZT row separates the global
headline scaler from outer-training-fold neural mean/SD; the vapor row separates
the fixed temporal `T_r=362.72 K` from outer-training-fold development medians
and correctly says three physical-pressure seed predictions. Method and
Limitations now say the paper displays cross-chart response discrepancy and
retains projection residuals in artifacts.

The repaired official build remains 13 US-Letter pages with main text ending on
page 9. It has no undefined citation/reference, overfull box, LaTeX/BibTeX
error, fatal error or rerun warning; Appendix Table 3 was visually rechecked and
remains readable. Current PDF, Experiments, protocol appendix, Method and
Limitations SHA-256 values are
`602ceeae1eb99bdd65a7ed6d874f19d1d402220af1ce3ef7dc8ef35b56dc1f17`,
`41fe8d4223a1a73a6262c2285b8d25d3026646dab6a902aada298e0bbffb5e65`,
`0ad483739d39e33475c3e977811dad6ca30922a74bc06ed2368f7817060c7abd`,
`6eccd9b3326724cc344081431711afd2690318d51270b67e6c6fb736d741f346`,
and `825ceb50446fd1ae5d7c8269416e8189eb56a3fe3846015099ebb0875b8da559`.
A fresh zero-context Run07 is active against this repaired state; no canonical
PASS is claimed before it completes.

## 153. Crystal guarded matrix resumes on four released GPUs (2026-08-29 22:12 CST)

At 22:12 CST the authoritative launcher advanced from its resource wait and
started fold 3 seeds 1--4 on GPUs 2--5.  The live state is now
`16 completed / 4 active / 5 pending / 0 failures`, with fold 3 seeds 1--4
bound respectively to GPUs 2, 3, 4 and 5.  Host NVML reports approximately
799 MiB used on each of those cards immediately after launch; the launcher's
`maximum_used_mib=1024` remains only the pre-launch availability guard, not a
training-memory cap.  `confirmation_targets_opened=false`, and no active-cell
metric was inspected.  The terminal all-development watcher remains separate
and cannot proceed before a clean 25/25 development state.

After warm-up, host NVML at 22:42 CST showed GPUs 2--5 using approximately
74.1/76.3/63.5/77.1 GiB-equivalent MiB readings at 99--100% utilization.  This
directly confirms that the 1024 MiB guard does not cap training allocation.

Run07 has frozen 118 semantic paper/evidence inputs plus an explicitly volatile,
embedded launcher-status snapshot used only to prove scope and confirmation
non-access.  Its fresh reviewer and independent figure check are still active;
the figure check found no numerical/hash mismatch and only two maintenance or
wording observations, which remain provisional until full Run07 reconciliation.

## 154. Current citation verdict is semantically useful but hash-stale (2026-08-29 22:36 CST)

The existing canonical `paper/CITATION_AUDIT.json` records a provisional PASS
for all 21 cited entries, and the bibliography hash still matches.  It cannot be
treated as a current strict artifact, however: its audited Introduction and
Experiments hashes are `b78d188e...` and `56b21526...`, whereas the current
files hash to `377b7c24...` and `41fe8d42...`; its main-entry key also names the
earlier `iclr_draft.tex` rather than the current `iclr2027_draft.tex` entry.
Related Work and the bibliography remain hash-matched.  This is snapshot
staleness, not evidence that a citation is false.  Do not use the old PASS to
erase Run07's literature/context `UNVERIFIABLE` scope.  After terminal Crystal
integration freezes the paper, run a fresh citation existence/context audit and
emit a new canonical artifact bound to the current recursive manuscript before
submission-readiness or top-50 completion is claimed.

## 155. Run07 closes scientific-data mismatches but remains provisional WARN (2026-08-29 22:38 CST)

Run07 is fully archived at
`paper/.aris/traces/paper-claim-audit/2026-08-29_run07/`.  The base fresh
reviewer reported `89 PASS / 13 UNVERIFIABLE / 0 FAIL` over 102 claims.  Its
only scientific-data non-pass, C077, resulted from overlooking the already
frozen current-v4 `router_margin` `entity_metrics.csv`.  Evidence-only
reconciliation against that file and the current runner/analyzer independently
reproduced 247 entities, 238 passes, NRMSE median/p95/max
`0.029577814373804807 / 0.21186416976145642 / 1.7449515101552815` and 748
negative predictions, so reconciled counts are
`90 PASS / 12 UNVERIFIABLE / 0 WARN / 0 FAIL`.  The original WARN and the
continuation reconciliation are retained separately; the continuation is not
misrepresented as a new independent PASS.

The remaining 12 items are nine literature/context/novelty claims and three
AI/process attestations outside the frozen raw-experiment evidence boundary.
All six Run06 protocol repairs and all three figures pass, with no demonstrated
scientific-data mismatch.  Assurance therefore remains overall `WARN`,
`same-family/provisional`, not submission-ready.  The 137 semantic inputs have
zero terminal hash drift and zero manifest mismatch; JSON/JSONL and the
102-entry ledger validate.  Manifest, RESPONSE, ledger and event-log SHA-256
values are `ccadaea09d7bf0cf972404543aaf542e83a3d023f4fe72d7a2aea65e6acfe05f`,
`c73bedd9ab7b91617923557027c46d70a9a9bcb7c961ebc4800038db6e85e9aa`,
`c7495660960c3b6382dd3adc2f50a6b9457857f437cc26554bd03ef3aeea5436`
and `5c75daa2e1b06bf4bd73454d96eed592d5b50e9abc12d4a83bc8dd6378f4b1a2`.
Run08 is a genuinely new zero-context audit and remains active; it, rather than
the Run07 continuation, is the next independent paper-to-evidence gate.

## 156. Crystal development reaches the final cell (2026-08-30 04:17 CST)

The authoritative launcher now reports `24 completed / 1 active / 0 pending /
0 failures`.  The sole remaining cell is fold 4 seed 4 on GPU 3; folds 0--3 and
fold 4 seeds 0--3 are terminal successes.  `confirmation_targets_opened=false`.
Host NVML shows only the final cell's initial approximately 799 MiB allocation
on GPU 3 at the snapshot, while GPUs 2, 4 and 5 are idle.  The development and
all-development watcher processes are both still alive.  Do not aggregate,
open confirmation, or start the all-development components early: the frozen
gate requires a clean `completed_all` 25/25 state first.  No active-cell metric
was inspected.

The Run08 worker was no longer live after the overnight continuation and had
left no trace directory, so it was explicitly resumed with instructions to
recover honestly or declare lost context rather than fabricate an archive.  A
Run08 result is not claimed until its complete trace and hashes exist.

## 157. Run08 independently confirms zero scientific mismatch (2026-08-30 04:27 CST)

Run08 recovered honestly and is complete at
`paper/.aris/traces/paper-claim-audit/2026-08-29_run08/`.  It independently
audited 147 declared/current files and 79 deduplicated claims, with
`43 exact_match / 33 rounding_ok / 3 missing_evidence` and zero material
scientific-data, configuration, aggregation or scope mismatches.  The three
unverified groups are literature/bibliography context, complete AI/process
provenance, and official-template provenance; these require their separate
integrity gates and do not negate the raw scientific reconciliation.  Run08 is
still `WARN`, `same-family/provisional`, not submission-ready assurance.

Manifest, proposed canonical JSON, claim-ledger and audited-input-map SHA-256
values are `6e79939b35738ddb1af8f69604100c472df72b7a5f9a46aa4a0db5fe0946d87d`,
`b0be1c53abf5cc36bbf6274b7d493d779acee2fa7d3574137a042437edb84d70`,
`dbc04fad385b330991d24db44fba39c9a3d89c2704026e90244fa08d839faf3e`
and `e720ab2df2eadea28d6cd04a47ca5546d4666bfa88b537bc71bdf3392955c270`.
The proposed top-level canonical artifact must not yet be promoted: 146 hashes
remain fresh, but the live launcher-status path changes its timestamp while the
final cell runs.  Once the launcher reaches stable terminal status, refresh
only this scope snapshot/hash without reading cell metrics, or rerun the final
post-integration claim audit.  No top-level canonical, paper, code or result was
modified by Run08.

## 158. Crystal development matrix reaches a clean 25/25 terminal state (2026-08-30 04:59 CST)

The authoritative
`runs/thermoml_crystal_cp_neural_gird_development_20260829/launcher_status.json`
reached `state=completed_all` at filesystem timestamp 04:59:42 CST.  All 25
fold--seed cells are terminal successes, with `0 active / 0 pending / 0
failures`; `confirmation_targets_opened=false`.  No cell-level metric was
inspected during this transition.  The separate all-development watcher then
started the frozen aggregate analyzer as PID 2948757.  At the latest observation
the analyzer remained CPU-active and had not yet emitted a final decision file,
so this clean completion is not itself evidence that either the learned bridge
or GIRD gate passed.  Keep the one-shot confirmation sealed until the complete
aggregate decision and subsequent all-development package satisfy the frozen
orchestrator contract.

## 159. Frozen Crystal neural/GIRD aggregate fails; all-development fairness campaign starts (2026-08-30 06:30 CST)

The independent aggregate completed normally and atomically emitted
`analysis/decision.json`, `analysis/manifest.json`, and
`analysis/all_development_decision.json`.  All 475 declared input hashes and all
12 declared aggregate-output hashes independently pass `sha256sum -c`; the
aggregate decision and manifest SHA-256 values are
`f3615b74493726fd190af55b48d774d03249226a105f231ca192f024197d6d81`
and `c6d0159763f733437e96d1ee7b01c3cbaf0d711a2add0586e425137f87ea6e11`.
The prepared all-development decision SHA-256 is
`3709e9fc1838eb338d538d3ed0bfbc79f921f72e66124480de64ecdbca16ce7b`.
Coverage is exactly 25 cells / 5 folds / 5 seeds, and
`confirmation_targets_opened=false` throughout.

The frozen scientific decision is `learned_bridge_status=FAIL`,
`gird_status=FAIL`, and `conditional_gird_status=FAIL`.  This is not a leakage
failure: query-target perturbation passes, function-space cross-seed geometry
is substantially more stable than raw q (mean distance Spearman approximately
`0.9230` versus `0.6056`), and coefficients are finite.  The failure is
localized to transfer and identifiability.  Spread decoder-functional pooled
physical R2 is only `0.4655798661`; projection fidelity has median
`0.9966893` but minimum `0.0665238`; dictionary motifs recur in at most 1--2 of
5 folds for every seed; all 300 recomputed OMP margin certificates fail; and
the affine-gauge diagnostics fail with very large worst-case discrepancies.
In four-support entities, conditional GIRD median entity NRMSE is `0.0874195`,
worse than both direct-target dictionary `0.0496813` and lambda-zero/support
OMP `0.0537437`.  In the spread regime, selected GIRD equals lambda zero at
pooled R2 `0.6586317`, while direct support-structure re-q remains the much
stronger compact expression at `0.8593003`.

Interpret this as evidence that the learned decoder prior is not identifiable
under this heterogeneous sparse-support setting, not as evidence against the
response-coordinate symbolic interface or as grounds to exclude the dataset.
The frozen narrative therefore moves to Branch B: canonical response
coordinates and direct support structure are the main contribution; GIRD is a
transparent negative diagnostic and must not appear as a headline improvement.
The all-development fairness campaign nevertheless launched normally.  At the
06:30 snapshot it has four active tasks on GPUs 2--5 (`support_cnp`, `no_q`,
`neural_seed0`, `neural_seed1`), three pending neural seeds, zero failures, and
confirmation still sealed.  Do not inspect partial task metrics; wait for all
seven components and their formal seals.

At the 06:36 CST refresh, `no_q` became the first successful terminal task and
GPU 3 immediately started `neural_seed2`.  Campaign coverage is therefore
`1/7 completed / 4 active / 2 pending / 0 failures`; support CNP and neural
seeds 0--2 are active, neural seeds 3--4 remain pending, and confirmation is
still sealed.  No completed-task metric was inspected before the package seal.

## 160. Branch-B theory and manuscript integration requirements (2026-08-30 06:44 CST)

The current title and abstract are already compatible with Branch B and do not
headline GIRD.  Terminal integration should remain minimal and evidence-led:
replace the pending Crystal paragraph with the complete sealed registry; state
that `A`, `B1`, `B2`, and `H` describe response background, slope, curvature,
and support-domain boundary enhancement without claiming microscopic
parameters; convert the Related Work conditional GIRD sentence to the observed
diagnostic outcome; and state in Limitations that the learned prior did not add
value in the heterogeneous Crystal sparse-support setting.

One pre-result theoretical diagnostic is implemented and frozen but currently
under-explained in the manuscript.  The implementation clarification assigns
cross-seed/support-offset OMP certification to the aggregate and uses the
sufficient bound `4*||R||_F*epsilon + 2*epsilon^2` against each saved
winner--runner score margin.  The method text should state the narrow point:
linear response projection is continuous under the displayed singular-value
bound, whereas hard dictionary selection is certified stable only when its
margin clears the aggregate perturbation bound.  The observed `0/300`
certificates and 1--2/5 maximum motif recurrence then explain why stable
function-space geometry does not imply an identifiable learned symbolic prior.
Do not turn the sufficient certificate into a necessity claim, and do not
change the frozen atom library, epsilon, margin or decision threshold.

## 161. Crystal expression confirmation remains eligible, but evaluator implementation must wait (2026-08-30 06:52 CST)

Read-only reconciliation of
`THERMOML_CRYSTAL_CP_SINGLE_USE_CONFIRMATION_CONTRACT_20260829.{md,json}` and
`THERMOML_CRYSTAL_CP_JOINT_CONFIRMATION_SEAL_PLAN_20260829.md` shows that the
spread v4 expression endpoint is independent of GIRD promotion.  The failed
development learned-prior gate therefore forbids a GIRD headline but does not
cancel the unchanged expression endpoint: spread confirmation still tests
`gamma=100`, degree 2, inverse-square-root `delta=0.0003`, complete finite
support-only coverage, exact target invariance, and physical pooled R2 at least
0.85.  GIRD remains a controlled diagnostic regardless of any later metric.

The repository currently contains the frozen scientific/transaction contract,
joint-seal plan and synthetic transaction test, but no Crystal-specific joint
seal assembler, real single-use evaluator, or independent confirmation
analyzer.  This is an explicit remaining implementation step, not permission
to improvise.  Contract section 2 and the joint-plan execution order require
all neural/no-q/CNP/FPCA all-development component packages to be terminal
before the evaluator may be constructed.  Therefore do not implement or run it
while the current campaign is partial.  After the campaign seals all seven
tasks, the legal order is: implement and focused-test the target-blind evaluator
and analyzer without opening confirmation responses; assemble and independently
verify the fixed joint package; derive the sole consumed root from the joint
seal hash; write/fsync the exclusive consumed lock and started receipt before
the first response read; evaluate exactly once; independently analyze the
immutable outputs.  A post-receipt failure consumes the opportunity and never
permits a retry.

At the 06:58 CST campaign refresh, neural seeds 0 and 1 also reached successful
terminal states.  GPUs 4 and 5 immediately started neural seeds 3 and 4, so the
all-development campaign is now `3/7 completed / 4 active / 0 pending / 0
failures`.  Completed tasks are `no_q`, `neural_seed0`, and `neural_seed1`;
support CNP and neural seeds 2--4 are active.  All seven tasks have therefore
been launched, confirmation remains unopened, and no per-seed metric was
inspected before terminal component packaging.

At the 07:09 CST refresh, support CNP and neural seed 2 also completed
successfully.  Campaign coverage is now `5/7 completed / 2 active / 0 pending /
0 failures`; only neural seeds 3 and 4 remain active on GPUs 4 and 5.
Confirmation is still unopened and no component metric has been inspected.
GPUs 2 and 3 are released, but there is no remaining formal pending task to
place on them; do not introduce an unplanned experiment into the sealed chain.

## 162. Crystal all-development campaign is terminal after an entrypoint-only packaging recovery (2026-08-30 08:17 CST)

The authoritative campaign now reports `state=completed_all`, with all seven
tasks (`support_cnp`, `no_q`, and neural seeds 0--4) completed, no active or
pending tasks, no failures, and `confirmation_targets_opened=false`.  All four
formal component packages independently verify.  Their manifest SHA-256 values
are `6c60884ba9af2ca6a7c724de1a94ad7d2e5709253593418270d4cc513902cc93`
for support CNP, `ca03684ce9266eacc30fdafdbd66277d6a6a50957421f0a83c34564284dff59c`
for the no-q temperature MLP,
`4931952be797e212e1b1e39e089219fc77837c20ab8e7ca0959c58e26436bcd7`
for neural/GIRD, and
`768a1cdc319b446fb8d5334f4fbaa154fa0e94d9bccd32d70cac99955417c193`
for FPCA.

The original watcher completed all training, then exited because it invoked
the neural packager as a direct file and Python could not import `scripts`.
This was an orchestration-entrypoint defect, not a scientific or training
failure.  The frozen packager remained byte-identical at SHA-256
`ed8368bcad31ef77629b61dbc89099d7157d3edf4432fe520bc50e561523d508`;
11 focused launcher/packager tests passed, and the same source was executed via
`python -m scripts.package_thermoml_crystal_cp_all_development_neural_gird_20260829`.
The ledger records the failed direct entrypoint and this no-source-change
recovery before the terminal event.  The recovered neural seal retains the
frozen development decision SHA-256
`3709e9fc1838eb338d538d3ed0bfbc79f921f72e66124480de64ecdbca16ce7b`,
five seeds, 247 entities, 23,742 rows, and all three learned-prior claims as
`FAIL`.  This terminal transition now authorizes construction—but not yet
execution—of the target-blind single-use confirmation evaluator and analyzer
in the exact order recorded in section 161.

## 163. Crystal joint package is independently sealed; the one-shot confirmation remains unconsumed pending explicit approval (2026-08-30 09:02 CST)

The Crystal-specific joint assembler, independent verifier, single-use
evaluator and independent analyzer are now implemented at the corresponding
`scripts/*thermoml_crystal_cp*20260830.py` paths.  The evaluator preserves
negative heat-capacity predictions, evaluates v4 only in spread, covers all
three frozen regimes, reruns the `+1,000,000` redacted-target twin, and exposes
all 27 locked methods.  The independent analyzer recomputes coverage,
five-seed medians, physical metrics, entity/DOI bootstrap intervals, NRMSE
tails, negative counts, leakage and the expression gate.  Twenty-nine focused
tests passed, including a real-checkpoint CPU smoke of the entire locked
method/regime registry with calibration iterations shortened only inside the
test.

The fixed joint package is
`runs/thermoml_crystal_cp_joint_confirmation_package_20260829/`.  It binds 244
repository-relative files and 27 methods.  The joint manifest, inventory and
seal SHA-256 values are
`74130531077896feec071ced8fa100fd009a059fa2ff87a9626d61cefadca255`,
`0100b801e5d0aa4b077ad67b7b77b6c41c4dd0fbcaefe54f4fec3074321759d0`,
and `ed1a4f60294debad0ae927364ce03cdf32efe9a002b5345464ed99c00fe68b0f`.
The independent verification at
`runs/thermoml_crystal_cp_joint_confirmation_verification_20260830/` is
`PASS`: all component verifiers and all 244 inventory hashes pass,
`confirmation_source_members_opened=false`, and
`confirmation_targets_opened=false`.

Host NVML showed GPUs 2--5 idle at 4 MiB each.  A request to launch the sole
attempt on physical GPU 2 was rejected by the execution safety reviewer
because this specific irreversible, non-retryable confirmation consumption
needs explicit user approval even though general GPU experiments were already
authorized.  No workaround was attempted.  Read-only checks prove there is no
consumed output root, no launch log, no tmux session and no receipt; the sole
confirmation opportunity remains intact.  The next authorized action must be
an explicit user instruction acknowledging and approving consumption of the
single Crystal-Cp confirmation attempt.  Until then, do not launch, rename,
replace or rebuild the sealed package.

## 164. Branch-B terminal Crystal evidence is integrated into the 9-page ICLR draft (2026-08-30)

The manuscript now reports the completed Crystal development registry rather
than future-tense plans.  The main text gives the stage-level meanings of the
frozen expression coefficients (`A`: reference/background, `B1`: normalized
slope, `B2`: smooth curvature, `H`: support-diagnosed boundary enhancement)
without assigning microscopic or critical-parameter semantics.  It reports the
terminal 5-fold x 5-seed contrast: leakage passes and function-space geometry is
more reproducible than raw q (mean distance Spearman `0.9230` vs `0.6056`), but
the learned bridge and GIRD gates fail; spread decoder-functional physical R2 is
`0.4656`; on the consensus prior-eligible four-support subset,
conditional-GIRD median entity NRMSE is `0.0874`, worse than direct-target
dictionary `0.0497` and lambda-zero support OMP `0.0537` on the same subset.
The main narrative therefore treats GIRD as a falsifiable negative diagnostic
and keeps the independently selected response-coordinate expression
(`R2=0.8593`) as the Crystal contribution.

The predeclared OMP stability audit is now explained narrowly: failure of the
sufficient margin condition is not treated as necessary evidence of
instability.  Detailed terminal evidence is in the protocol appendix: median /
minimum projection R2 `0.9967/0.0665`, motif recurrence at most 1--2/5 folds per
seed, `0/300` certificate passes, and failed affine-gauge diagnostics.  The
Related Work and Limitations sections explicitly avoid promoting the failed
learned prior.

`scripts/build_iclr2027_paper.sh` completes successfully.  The resulting
`paper/iclr2027_draft.pdf` has 13 total pages; the main body now ends with the
Conclusion on page 9, while AI/reproducibility statements, references and the
appendix begin on page 10.  There are no undefined references/citations or
overfull boxes, and all PDF fonts are embedded.  The only remaining scientific
transition is the still-unconsumed single-use Crystal confirmation described in
section 163; manuscript edits did not touch the sealed evaluator, tests or joint
package.

Run09 zero-context paper-claim audit is archived at
`paper/.aris/traces/paper-claim-audit/2026-08-30_run09/`.  It audited 48 claims
against 329 hash-frozen permitted inputs without reading any Crystal
confirmation response/source member.  It found one material scope defect in an
intermediate draft: the four-support `0.0874/0.0497/0.0537` medians were
computed on the consensus prior-eligible subset, but the prose originally
omitted that population.  The final prose now names the subset twice; the fresh
reviewer reclassified the claim as `rounding_ok`.  No material blocker remains.
The follow-up wording fixes also replace ambiguous "learned rows" with
`CNP/no-q MLP use pointwise five-seed medians; other rows are deterministic`,
and replace an unsupported future "reports all tails" statement with the
seal-supported `One evaluation is authorized`.  Two nonblocking evidence
limits remain by design: the audit's permitted set did not include the delegated
v3 fitter for a direct explicit-SVD implementation check, and confirmation-era
/ InChIKey-disjointness claims cannot be reverified without crossing the
unopened-confirmation boundary.  Final post-wording hashes are
`f6fb9fa5480a31ce6df949ca0a24d91f8d893f055bd18f4fb6a698579d073437`
for `paper/sections/experiments.tex` and
`758c0fa766157700c659f265a58d3bf613dc96c20949e05d21769f05f5a4e14b`
for `paper/iclr2027_draft.pdf`; the main body still ends on page 9.

## 165. ICLR top-half gate and anonymous-supplement evidence close (2026-08-30 10:26 CST)

The final manuscript narrative was narrowed from a universal/canonical latent
variable claim to auditable, basis-relative response coordinates under a frozen
basis and probe specification.  A fresh same-family provisional reviewer now
judges the manuscript reasonably within the top half of the ICLR submission
pool, assigning a subjective probability of `55%` with interval `45--65%`.
The remaining rejection risk is novelty/impact rather than a demonstrated
scientific mismatch.  This assessment is archived in
`paper/ICLR_TOP50_FRESH_REVIEW_20260830.md`; it is not an acceptance forecast.

The official paper is `paper/iclr2027_draft.pdf`, SHA-256
`69b29b6d04375fd683482f7533bf3352ac74b3382551e9f713b5accfdde6ecfc`.
It has 13 pages, the main body ends on page 9, AI/reproducibility material starts
on page 10, all fonts are embedded, and the build has no undefined references,
undefined citations, or overfull boxes.  Citation freshness is recorded in
`paper/CITATION_AUDIT_FRESHNESS_20260830.{md,json}`: the same 21 keys and
semantically unchanged contexts remain covered by the prior citation audit.

The anonymous supplement is
`public/iclr2027_anonymous_supplement_20260830.tar.gz`, SHA-256
`484a5bf625e7d20e431b1f4f1a74512d8e63017f870c3cd4588dffdd5bc565ca`.
Its `SHA256SUMS` SHA-256 is
`656127bb533736ac0208bee14ff13437ed605f6cbbadd5f90163738d62817090`.
The persistent clean-room verifier extracted the archive under
`public/qa/iclr2027_supplement_cleanroom_20260830`, passed all 100 declared
file hashes, deleted the bundled PDF, rebuilt from that root, and reproduced
the official PDF byte-for-byte.  The receipt is
`public/qa/iclr2027_supplement_verification_20260830.json`, SHA-256
`73a4a5f6f137cac144bfde103c3306bb70933255696651536805753fce7ca157`.
Archive members use a deterministic noon timestamp that was about 97 minutes
ahead of the host clock during extraction; GNU tar emitted nonfatal future-time
warnings, while checksum verification and the causal rebuild still exited zero.

The narrow final claim re-review is `PASS (same-family provisional)`: all four
previously unsupported process claims are now `exact_match` (ZT a-priori
endpoint, verification from the supplement root, every bundled hash passing,
and byte-identical causal rebuild), with `0 unsupported / 0 mismatch`.
Previously audited scientific data, configuration and aggregation remain at
zero mismatch.  The record is
`paper/FINAL_INCREMENTAL_CLAIM_AUDIT_20260830.md`.

The sealed Crystal joint package is unchanged at manifest/inventory/seal hashes
`74130531077896feec071ced8fa100fd009a059fa2ff87a9626d61cefadca255`,
`0100b801e5d0aa4b077ad67b7b77b6c41c4dd0fbcaefe54f4fec3074321759d0`,
and `ed1a4f60294debad0ae927364ce03cdf32efe9a002b5345464ed99c00fe68b0f`.
No Crystal confirmation output root or receipt was created.  The one-shot
opportunity remains optional and unconsumed; do not launch it without the
user's explicit acknowledgement of that irreversible consumption.

The portable paper checkpoint was committed as `e0e8e87`
(`finalize auditable response-coordinate paper package`) and pushed to
`origin/research/latent-q-stagec-20260826` at 10:30 CST.  It contains the
official editable paper/PDF, citation and claim audits, vapor-selection audit,
deterministic anonymous supplement directory and tarball, clean-room receipt,
build/verifier scripts, and this handoff record.  It deliberately excludes
unrelated dirty-worktree changes, generated Python caches, local TeX logs, and
the still-unconsumed Crystal confirmation outputs.
