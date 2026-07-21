# TravelEval Partial Comparison Without Gaode

This is a no-Gaode partial comparison. FRTC/SSR/CSM and official route-distance metrics are skipped.

## Selected Cases

| Difficulty | UID | Bounded partial avg | BCS | CCD | FAR | VROH | TTCS | BE | EDI | ADS | AQE | Profit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | T0001 | 0.8013 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 27.2313 | 0.6466 | 0.565 | 0.9513 | 6.3667 |
| medium | T0201 | 0.739 | 1.0 | 0.0 | 0.2 | 0.0 | 0.0 | 7.2056 | 0.5749 | 0.3531 | 0.96 | 6.025 |
| hard | T0601 | 0.6522 | 1.0 | 0.0 | 0.625 | 0.0 | 0.0 | 6.6303 | 0.5962 | 0.452 | 0.94 | 4.7 |
| custom_easy_prompt | beijing_shanghai_original_prompt | 0.7086 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 6.9711 | 0.6372 | 0.4708 | 0.952 | 8.0 |

## Paper Comparison

| System | Bounded partial avg | BE | Profit |
|---|---:|---:|---:|
| Your pipeline, 4 run cases | 0.7253 | - | - |
| Claude Code | 0.6249 | 13.212 | 5.6632 |
| Approach A | 0.694 | 16.274 | 5.9979 |
| Approach B | 0.7274 | 22.192 | 5.5015 |

Bounded partial avg converts lower-is-better metrics with `1 - value`, keeps higher-is-better metrics as-is, clips to 0..1, and excludes Gaode-required metrics plus unbounded cost decomposition fields.
