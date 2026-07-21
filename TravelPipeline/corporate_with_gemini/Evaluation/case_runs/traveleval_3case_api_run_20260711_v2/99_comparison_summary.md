# TravelEval Partial Comparison Without Gaode

This is a no-Gaode partial comparison. FRTC/SSR/CSM and official route-distance metrics are skipped.

## Selected Cases

| Difficulty | UID | Bounded partial avg | BCS | CCD | FAR | VROH | TTCS | BE | EDI | ADS | AQE | Profit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | T0001 | 0.7744 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 11.6167 | 0.5382 | 0.4708 | 0.9608 | 7.225 |
| medium | T0201 | 0.7519 | 1.0 | 0.0 | 0.2 | 0.25 | 0.0 | 7.2052 | 0.6499 | 0.3531 | 0.9352 | 5.9 |
| hard | T0601 | 0.6026 | 0.0 | 0.0 | 0.6667 | 0.0 | 0.0 | 0.746 | 0.4362 | 0.1695 | 0.96 | 4.8 |

## Paper Comparison

| System | Bounded partial avg | BE | Profit |
|---|---:|---:|---:|
| Your pipeline, selected 3 cases | 0.7096 | - | - |
| Claude Code | 0.6249 | 13.212 | 5.6632 |
| Approach A | 0.694 | 16.274 | 5.9979 |
| Approach B | 0.7274 | 22.192 | 5.5015 |

Bounded partial avg converts lower-is-better metrics with `1 - value`, keeps higher-is-better metrics as-is, clips to 0..1, and excludes Gaode-required metrics plus unbounded cost decomposition fields.
