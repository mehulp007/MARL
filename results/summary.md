# Evaluation summary

Mean time in system (seconds; lower is better). Every controller sees the same vehicles.

| Scenario | cooperative | selfish | maxpressure | fixed | random |
|---|---|---|---|---|---|
| calm_low | 116.4 | 113.8 | 128.7 | 162.9 | 447.4 |
| steady_medium | 150.9 | 148.0 | 164.2 | 197.9 | 619.5 |
| heavy_high | 272.5 | 261.2 | 277.3 | 296.0 | 694.5 |
| rush_hour_wave | 185.1 | 178.7 | 199.3 | 227.2 | 634.2 |
| arterial_incident | 150.6 | 147.3 | 164.2 | 197.9 | 619.4 |
| center_closure | 168.7 | 160.8 | 174.5 | 214.0 | 626.2 |
| stadium_surge | 190.8 | 186.2 | 200.3 | 236.9 | 658.7 |
| emergency_corridor | 150.3 | 147.3 | 163.3 | 196.4 | 616.3 |
| compound_chaos | 255.5 | 249.1 | 264.8 | 270.6 | 677.3 |

## Arterial Green Wave Score (higher is better)

| Scenario | cooperative | selfish | maxpressure | fixed | random |
|---|---|---|---|---|---|
| calm_low | 0.70 | 0.72 | 0.62 | 0.63 | 0.11 |
| steady_medium | 0.55 | 0.56 | 0.43 | 0.55 | 0.10 |
| heavy_high | 0.34 | 0.36 | 0.25 | 0.41 | 0.11 |
| rush_hour_wave | 0.47 | 0.49 | 0.35 | 0.50 | 0.10 |
| arterial_incident | 0.55 | 0.57 | 0.43 | 0.55 | 0.10 |
| center_closure | 0.49 | 0.52 | 0.41 | 0.54 | 0.10 |
| stadium_surge | 0.47 | 0.49 | 0.37 | 0.53 | 0.10 |
| emergency_corridor | 0.55 | 0.56 | 0.43 | 0.56 | 0.10 |
| compound_chaos | 0.39 | 0.41 | 0.31 | 0.48 | 0.10 |

## Emergency-vehicle stops per trip (lower is better)

| Scenario | cooperative | selfish | maxpressure | fixed | random |
|---|---|---|---|---|---|
| calm_low | 0.14 | 0.06 | 2.25 | 1.90 | 8.07 |
| steady_medium | 0.22 | 0.15 | 2.36 | 2.10 | 14.40 |
| heavy_high | 1.72 | 1.54 | 4.09 | 3.24 | 9.19 |
| rush_hour_wave | 0.48 | 0.52 | 2.35 | 1.94 | 8.27 |
| arterial_incident | 0.29 | 0.19 | 2.43 | 2.10 | 14.50 |
| center_closure | 0.62 | 0.48 | 2.94 | 2.20 | 15.13 |
| stadium_surge | 1.23 | 1.30 | 4.23 | 2.98 | 8.80 |
| emergency_corridor | 0.26 | 0.17 | 2.20 | 1.94 | 14.68 |
| compound_chaos | 3.35 | 3.85 | 6.90 | 3.07 | 14.50 |

## Typical longest wait at a junction, seconds (lower is better)

| Scenario | cooperative | selfish | maxpressure | fixed | random |
|---|---|---|---|---|---|
| calm_low | 25 | 24 | 54 | 48 | 169 |
| steady_medium | 64 | 70 | 75 | 93 | 236 |
| heavy_high | 117 | 133 | 93 | 152 | 250 |
| rush_hour_wave | 82 | 87 | 79 | 106 | 231 |
| arterial_incident | 64 | 68 | 76 | 93 | 236 |
| center_closure | 75 | 77 | 76 | 124 | 242 |
| stadium_surge | 78 | 84 | 79 | 108 | 236 |
| emergency_corridor | 66 | 69 | 75 | 101 | 236 |
| compound_chaos | 113 | 120 | 92 | 151 | 250 |

## What each perturbation cost

Extra queueing caused by the events, in vehicle-hours: each episode is compared with the same seed run without the events. In brackets: seconds until the extra queue is gone.

| Scenario | cooperative | selfish | maxpressure | fixed | random |
|---|---|---|---|---|---|
| arterial_incident | -0.3 (241 s) | -0.6 (263 s) | -0.2 (349 s) | 0.0 (0 s) | -0.1 (0 s) |
| center_closure | 13.9 (1251 s) | 9.5 (1044 s) | 7.3 (1243 s) | 13.3 (964 s) | 6.8 (233 s) |
| stadium_surge | 27.6 (1500 s) | 25.5 (1500 s) | 26.9 (1473 s) | 27.1 (1500 s) | 0.3 (0 s) |
| compound_chaos | 63.0 (1124 s) | 65.2 (1406 s) | 64.2 (1309 s) | 35.9 (1600 s) | 13.5 (963 s) |

## Acceptance gates

- **G1 PASSED** - cooperative time in system at least 15% lower than the fixed timer (medium demand)
- **G2 PASSED** - cooperative time in system within 5% of max-pressure, or better (medium demand)
- **G3 NOT MET** - gws_arterial: cooperative > selfish > random (medium demand)
- **G4 NOT MET** - the incident and the surge cause less extra queueing under cooperative than under the fixed timer (same seed with and without the event)
- **G5 PASSED** - emergency vehicles stop less often under cooperative than under the fixed timer

## Cooperative versus selfish (medium demand, per training seed)

- cooperative: cooperative_s0 150.3, cooperative_s1 152.2, cooperative_s2 150.4 (mean 150.9)
- selfish: selfish_s0 146.3, selfish_s1 148.3, selfish_s2 149.3 (mean 148.0)
- per-seed ranges overlap: False
