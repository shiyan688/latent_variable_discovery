# Protocol-matched functional-coordinate diagnostic

This is a post-terminal diagnostic of saved q candidates; it does not change the frozen selection or authorize validation.
For each battery, the probe uses the protocol of its first observed discharge and evaluates cycles 1, 10, 20, and 28.
No target value is used to choose the protocol.

## Fixed-probe support

| inner split | exact rows | fraction | labels covered |
|---|---|---|---|
| nasa_battery_capacity_reviewer_clean_inner0 | 168/711 | 0.236 | 1/8 |
| nasa_battery_capacity_reviewer_clean_inner1 | 0/716 | 0.000 | 0/8 |
| nasa_battery_capacity_reviewer_clean_inner2 | 168/711 | 0.236 | 1/8 |

## Cross-seed rank stability at observed protocols

| weight | coordinate | median of splits | worst seed-pair/split |
|---|---|---|---|
| 0 | matched_capacity_cycle1 | 1.000 | 0.976 |
| 0 | matched_early_fade_rate | 0.869 | 0.452 |
| 0.001 | matched_capacity_cycle1 | 1.000 | 0.976 |
| 0.001 | matched_early_fade_rate | 0.869 | 0.452 |
| 0.01 | matched_capacity_cycle1 | 1.000 | 0.976 |
| 0.01 | matched_early_fade_rate | 0.821 | 0.381 |
| 0.1 | matched_capacity_cycle1 | 1.000 | 0.976 |
| 0.1 | matched_early_fade_rate | 0.845 | 0.452 |
| 1 | matched_capacity_cycle1 | 1.000 | 0.976 |
| 1 | matched_early_fade_rate | 0.893 | 0.476 |

## Alignment with empirical early-curve descriptors

These correlations are post-hoc development diagnostics and use meta-fit target values; they are not selection gates.

| weight | capacity Spearman | early-fade Spearman |
|---|---|---|
| 0 | 0.976 | 0.786 |
| 0.001 | 0.976 | 0.786 |
| 0.01 | 0.976 | 0.738 |
| 0.1 | 0.976 | 0.714 |
| 1 | 0.976 | 0.690 |

Interpretation must distinguish a failure of a fixed off-protocol coordinate from a failure of the learned response geometry.
