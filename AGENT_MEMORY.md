---
document_type: agent_handoff_memory
project: latent_variable_search
last_updated: 2026-08-26 21:55 CST
current_branch: research/latent-q-stagec-20260826
base_commit: 2b13869
live_status_source: runs/nasa_battery_reviewer_clean_inner_symbolic_20260825/status.json
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

`scripts/run_iclr_real_discovery.py` now accepts explicit `--support-split-mode prefix --support-order-column discharge_index`, passes the same prefix rule into q calibration, and saves both `train_label_q.csv` and `training_checkpoint.pt`. A two-epoch CPU smoke completed at `/tmp/lvs_nasa_clean_smoke`: 13 train q rows, 5 test q rows, a loadable checkpoint, 132 support rows, and 312 query rows. For every outer battery the last support cycle is strictly before the first query cycle. The smoke metrics are structural only and must never be quoted scientifically.

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

## 40. Active meta-selected soft q-prior diagnostic (2026-08-26 21:46 CST)

The next protocol is frozen in `NASA_META_SELECTED_Q_PRIOR_PLAN_20260826.md` using the academic-research-suite experiment-agent boundary. For each prefix-q checkpoint it scores the fixed grid `{0, 0.001, 0.01, 0.1, 1}` by leave-one-entity-out calibration on all eight meta-fit entities. Selection uses only the calibration-internal holdout inside each entity's earliest 30% support; later meta-fit targets and all structure-validation query targets are excluded. The eight selected support-calibrated meta-fit q values then define the prior population for the five structure-validation calibrations. A +123.456 query perturbation audits that selected q is unchanged.

Advancement retains the prediction/interface/stability gates from the prefix-q pilot. This stage tests whether a support-only selected soft standardized prior can remove the large calibration tails without the prediction loss of convex/box projection. It remains sequential inner-split development, not independent confirmation.

Static compilation, CLI checks, `git diff --check`, and 57 tests passed. At 21:44 CST no GPU was empty: physical 0/7 had new ~72.5 GiB VLLM workers, 6 had a ~24.9 GiB sglang worker, and 1--5 were also occupied by unrelated jobs. The formal root remained absent and no formal process was launched. A non-counted CPU smoke first exposed a summary-only DataFrame selection bug; the minimal fix changed functional-shift calculation to use the already merged selected-q table. The rerun passed: 5/5 weights scored, 8/5 entities, zero query leakage, selected weight 0, raw/functional max-|z| 3.985/1.078, and selected validation NRMSE 1.3661 versus 1.3703 prefix-q no-prior. These values are structural/early observations only. Formal execution must wait for a fresh empty-card check and must not claim a utilization-zero but memory-occupied card.

At 21:50 CST a second host-level snapshot still found no empty card: GPUs 0/1/7 each used about 72.4--72.6 GiB, GPUs 2/3 about 70.9 GiB, GPUs 4/5 about 55.8 GiB, and GPU 6 about 24.9 GiB. The utilization-zero readings on 0/1/6/7 do not make those cards available. No formal prior cell or waiting controller was started. The beginner-readable main report now contains a four-row causal comparison of support matching, convex bounding, coordinate-box bounding, and prefix-q training; compact terminal summaries for all four completed stages are explicitly included by `.gitignore` while raw checkpoints, predictions, and logs remain local.

The verified transition was committed as `e8d4f5c` (`diagnose and align NASA latent q interface`) and pushed to `origin/research/latent-q-stagec-20260826` at 21:52 CST. The pre-commit suite was 57 passed with only the two expected small-sample R-squared warnings. A fresh agent should not rerun the four terminal diagnostics; it should refresh `nvidia-smi`, confirm `runs/nasa_meta_selected_q_prior_20260826/` is absent or reconcile any new raw cells, and then execute the already frozen formal prior commands only on genuinely empty cards.
