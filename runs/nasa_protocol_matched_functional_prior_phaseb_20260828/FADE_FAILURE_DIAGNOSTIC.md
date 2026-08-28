# Phase-B early-fade failure diagnostic

This is a post-terminal diagnostic. It does not change the frozen Phase-B STOP decision or authorize Stage C2.

The selected prior retained prediction, capacity, empirical alignment, and the full four-response geometry. The only failed frozen gate is early-fade cross-seed stability on inner2 (split median 0.20); weight 0 has the same 0.20, so the failure is not introduced by weight 0.01.

## Difference-coordinate noise

| split | fade seed SD / entity spread | capacity seed SD / entity spread |
|---|---|---|
| nasa_battery_capacity_reviewer_clean_inner0 | 0.502 | 0.032 |
| nasa_battery_capacity_reviewer_clean_inner1 | 0.362 | 0.041 |
| nasa_battery_capacity_reviewer_clean_inner2 | 0.585 | 0.037 |

Early fade is a small difference between two large decoder outputs. Across splits, seed variation is 36--58% of between-battery fade spread, versus only 3--4% for capacity. Thus stable four-point response distances can coexist with an unstable derivative-like rank.

## Input-series conditions that violate a simple early-degradation interpretation

| split | battery | first/following capacity | protocols through cycle 28 |
|---|---|---|---|
| nasa_battery_capacity_reviewer_clean_inner0 | B0036 | 0.559 | 1 |
| nasa_battery_capacity_reviewer_clean_inner0 | B0039 | 0.255 | 2 |
| nasa_battery_capacity_reviewer_clean_inner1 | B0033 | 0.525 | 1 |
| nasa_battery_capacity_reviewer_clean_inner1 | B0040 | 0.871 | 2 |
| nasa_battery_capacity_reviewer_clean_inner2 | B0036 | 0.559 | 1 |
| nasa_battery_capacity_reviewer_clean_inner2 | B0039 | 0.255 | 2 |

B0036, B0039, and B0033 begin with large recovery/activation transients rather than monotone fade; B0039/B0040 also change operating protocol within 28 cycles. A cycle-1-to-10 slope is therefore not uniformly the same physical estimand across these batteries.

## Mechanistic consequence

Rank 2 intentionally leaves both dominant capacity and response-shape directions unpenalized. It preserves legitimate response geometry but cannot canonicalize a noisy fade direction inside that retained subspace. The next justified screen is rank 1 at protocol-matched probes: preserve the dominant capacity direction and softly regularize residual response shape. This must be treated as sequential development and later confirmed on new batteries.
