### Part 1: incremental vs cold-start SAT (ba32_4500, F_priv k=1 S=2, first exact rule)

| L | cold step s | cold sum over 5..L s | incr_keep_step1 step s / cum s | incr_prev_5-10-20-40 step s / cum s | incr_prev_step1 step s / cum s |
|---|---|---|---|---|---|
| 5 | 16.11 (SAT) | 16.1 | 4.87 / 15.97 (SAT) | 15.74 / 15.74 (SAT) | 5.75 / 16.37 (SAT) |
| 10 | 524.98 (SAT) | 541.1 | 263.3 / 819.62 (SAT) | UNKNOWN_timeout | 850.57 / 1243.0 (SAT) |
| 20 | UNKNOWN_timeout | UNKNOWN | UNKNOWN_timeout | UNKNOWN_timeout | UNKNOWN_timeout |
| 40 | UNKNOWN_timeout | UNKNOWN | UNKNOWN_timeout | UNKNOWN_timeout | UNKNOWN_timeout |

longest length reached per incremental arm: {'incr_keep_step1': 12, 'incr_prev_5-10-20-40': 5, 'incr_prev_step1': 12}
peak RSS GB: {'cold_keep_L10': 5.32, 'cold_keep_L20': None, 'cold_keep_L40': None, 'cold_keep_L5': 1.96, 'incr_keep_step1': 5.19, 'incr_prev_5-10-20-40': 1.95, 'incr_prev_step1': 5.85}

### Part 2: SAT repair from different starts (warm start = phase preference + Hamming objective, NOT a hard constraint)

| clip | start | status | first exact s (encode s) | repair distance at first exact | best repair-distance UPPER BOUND (s) | keep-exc rule bits | Node exact | GPU cost of start |
|---|---|---|---|---|---|---|---|---|
| ba32_3000_5 | BP0.1 | BEST_SO_FAR_budget | 13.2 (11.9) | 239 | (83, 440.1) | 1117 | True | 614 GPU-s train+compile (4 seeds vmapped); selected ckpt at 40.1 s |
| ba32_3000_5 | BP1 | BEST_SO_FAR_budget | 13.1 (11.8) | 234 | (96, 503.6) | 1104 | True | 614 GPU-s train+compile (4 seeds vmapped); selected ckpt at 40.0 s |
| ba32_3000_5 | R0 | BEST_SO_FAR_budget | 14.8 (11.9) | 282 | (82, 540.0) | 1722 | True | 607 GPU-s train+compile (4 seeds vmapped); selected ckpt at 100.1 s |
| ba32_3000_5 | conway | BEST_SO_FAR_budget | 13.4 (11.6) | 340 | (183, 288.0) | 1449 | True | - |
| ba32_3000_5 | keep | BEST_SO_FAR_budget | 13.0 (11.8) | 237 | (99, 503.1) | 715 | True | - |
| ba32_3000_5 | random | BEST_SO_FAR_budget | 15.4 (11.9) | 333 | (190, 248.8) | 2346 | True | - |
| ba32_4500_10 | R0 | BEST_SO_FAR_budget | 1671.7 (26.8) | 523 | (523, 1671.7) | 2173 | True | 608 GPU-s train+compile (4 seeds vmapped); selected ckpt at 240.4 s |
| ba32_4500_10 | conway | UNKNOWN_timeout_no_solution | - (None) | - | - | - | - | - |
| ba32_4500_10 | keep | BEST_SO_FAR_budget | 518.2 (26.5) | 573 | (572, 813.7) | 1940 | True | - |
| ba32_4500_10 | random | BEST_SO_FAR_budget | 1562.5 (26.5) | 724 | (724, 1562.5) | 2258 | True | - |

### Part 3: exact rules -> circuits (bits include gate ops, wiring, outputs, constants)

| rule | target | consulted | full: LUT-exc bits | full: best circuit bits (gates, flow) | reach: LUT-exc bits | reach: best circuit bits (gates, flow) | best recording | all checks |
|---|---|---|---|---|---|---|---|---|
| exp03/ba32_3000_5_E_keep.min.rule.json | ba32_3000_5 | 314 | 1992 | 4375 (237, abc_default) | 1187 | 4515 (244, abc_x4) | 195 | True |
| exp03/ba32_3000_5_E_keep.rule.json | ba32_3000_5 | 314 | 1187 | 7021 (358, keepxor_abc_x4) | 1187 | 4709 (253, abc_x4) | 195 | True |
| exp03/ba32_3000_5_E.min.rule.json | ba32_3000_5 | 495 | 2027 | 6295 (325, abc_x4) | 1662 | 6845 (350, abc_x4) | 195 | True |
| exp03/ba32_3000_5_E.rule.json | ba32_3000_5 | 495 | 2051 | 14319 (674, keepxor_abc_x4) | 1662 | 6625 (340, abc_x4) | 195 | True |
| exp03/ba32_4500_5_E_keep.min.rule.json | ba32_4500_5 | 625 | 2180 | 10671 (522, abc_x4) | 1899 | 11055 (538, abc_x4) | 678 | True |
| exp03/ba32_4500_5_E_keep.rule.json | ba32_4500_5 | 625 | 1899 | 14775 (693, keepxor_abc_x4) | 1899 | 11007 (536, abc_x4) | 678 | True |
| exp03/ba32_4500_5_E.min.rule.json | ba32_4500_5 | 544 | 2150 | 9155 (455, abc_x4) | 1808 | 8671 (433, keepxor_abc_x4) | 678 | True |
| exp03/ba32_4500_5_E.rule.json | ba32_4500_5 | 544 | 1995 | 12207 (586, abc_x4) | 1808 | 8781 (438, abc_x4) | 678 | True |
| exp03/bar12_X6_keep.min.rule.json | bar12 | 819 | 2317 | 17081 (789, abc_x4) | 2187 | 16241 (754, keepxor_abc_x4) | 825 | True |
| exp03/bar12_X6_keep.rule.json | bar12 | 819 | 2187 | 19673 (897, keepxor_abc_x4) | 2187 | 16241 (754, keepxor_abc_x4) | 825 | True |
| exp03/bar12_X6.min.rule.json | bar12 | 739 | 2289 | 15185 (710, abc_x4) | 2136 | 14081 (664, keepxor_abc_x4) | 825 | True |
| exp03/bar12_X6.rule.json | bar12 | 739 | 2350 | 20657 (938, abc_x4) | 2136 | 14081 (664, keepxor_abc_x4) | 825 | True |
| exp03c/ba32_3000_5_E.rule.json | ba32_3000_5 | 193 | 588 | 2995 (168, keepxor_abc_x4) | 588 | 2133 (125, keepxor_abc_x4) | 195 | True |
| exp03c/ba32_4500_5_E.rule.json | ba32_4500_5 | 422 | 1159 | 7373 (374, keepxor_abc_x4) | 1159 | 5657 (296, keepxor_abc_x4) | 678 | True |
| exp03c/bar12_X6.rule.json | bar12 | 762 | 2109 | 18377 (843, keepxor_abc_x4) | 2109 | 14465 (680, keepxor_abc_x4) | 825 | True |
| exp03c/disc12_E.rule.json | disc12 | 312 | 1143 | 6933 (354, keepxor_abc_x4) | 1143 | 4975 (265, keepxor_abc_x4) | 554 | True |
| exp03c/disc12_X1.rule.json | disc12 | 403 | 2132 | 12469 (596, keepxor_abc_x4) | 2132 | 9350 (463, keepxor_abc_x4) | 554 | True |
| exp03/disc12_E_keep.min.rule.json | disc12 | 396 | 2105 | 6911 (353, abc_x4) | 1495 | 6801 (348, abc_x4) | 554 | True |
| exp03/disc12_E_keep.rule.json | disc12 | 396 | 1495 | 10261 (505, keepxor_abc_x4) | 1495 | 6801 (348, abc_x4) | 554 | True |
| exp03/disc12_E.min.rule.json | disc12 | 572 | 2319 | 10407 (511, abc_x4) | 1935 | 10815 (528, keepxor_abc_x4) | 554 | True |
| exp03/disc12_E.rule.json | disc12 | 572 | 2336 | 18615 (853, abc_x4) | 1935 | 10599 (519, abc_x4) | 554 | True |
| exp03/disc12_X1_keep.min.rule.json | disc12 | 516 | 5861 | 10098 (497, abc_x4) | 3103 | 10355 (508, keepxor_abc_x4) | 554 | True |
| exp03/disc12_X1_keep.rule.json | disc12 | 516 | 3103 | 18997 (868, keepxor_abc_x4) | 3103 | 10355 (508, keepxor_abc_x4) | 554 | True |
| exp03/disc12_X1.min.rule.json | disc12 | 420 | 5908 | 8338 (417, abc_x4) | 3081 | 9108 (452, keepxor_abc_x4) | 554 | True |
| exp03/disc12_X1.rule.json | disc12 | 420 | 4870 | 15013 (702, abc_x4) | 3081 | 9108 (452, keepxor_abc_x4) | 554 | True |
| exp03/disc12_X2_keep.min.rule.json | disc12 | 523 | 2243 | 10057 (496, abc_x4) | 1804 | 9045 (450, keepxor_abc_x4) | 554 | True |
| exp03/disc12_X2_keep.rule.json | disc12 | 523 | 1804 | 14487 (681, keepxor_abc_x4) | 1804 | 9639 (477, keepxor_abc_x4) | 554 | True |
| exp03/disc12_X2.min.rule.json | disc12 | 540 | 2253 | 9793 (484, abc_x4) | 1897 | 10057 (496, keepxor_abc_x4) | 554 | True |
| exp03/disc12_X2.rule.json | disc12 | 540 | 2226 | 16599 (769, abc_x4) | 1897 | 10057 (496, keepxor_abc_x4) | 554 | True |
| exp06/ba32_4500_10_E.rule.json | ba32_4500_10 | 651 | 1940 | 15279 (714, keepxor_abc_x4) | 1940 | 10791 (527, keepxor_abc_x4) | 1260 | True |
| exp06/ba32_4500_5_E.rule.json | ba32_4500_5 | 422 | 1159 | 7835 (395, keepxor_abc_x4) | 1159 | 5657 (296, keepxor_abc_x4) | 678 | True |
| exp07/D2_main/ba32_3000_5_E.rule.json | ba32_3000_5 | 297 | 1726 | 5217 (276, abc_x4) | 1056 | 2795 (158, abc_x4) | 195 | True |
| exp07/T2_main/best.rule.json | ba32_3000_5 | 314 | 2155 | 11247 (546, abc_x4) | 1187 | 4709 (253, abc_x4) | 195 | True |
| ../../exp09/ba32_4500_E_cold_keep_L10.L10.rule.json | ba32_4500_10 | 629 | 1941 | 13311 (632, keepxor_abc_x4) | 1941 | 10101 (498, keepxor_abc_x4) | 1260 | True |
| ../../exp09/ba32_4500_E_cold_keep_L5.L5.rule.json | ba32_4500_5 | 625 | 1899 | 14775 (693, keepxor_abc_x4) | 1899 | 11007 (536, abc_x4) | 678 | True |
| ../../exp09/ba32_4500_E_incr_keep_step1.L10.rule.json | ba32_4500_10 | 601 | 1859 | 15183 (710, keepxor_abc_x4) | 1859 | 10123 (499, keepxor_abc_x4) | 1260 | True |
| ../../exp09/ba32_4500_E_incr_keep_step1.L5.rule.json | ba32_4500_5 | 419 | 1548 | 11487 (556, keepxor_abc_x4) | 1548 | 7153 (364, keepxor_abc_x4) | 678 | True |
| ../../exp09/ba32_4500_E_incr_prev_5-10-20-40.L5.rule.json | ba32_4500_5 | 625 | 1899 | 14487 (681, keepxor_abc_x4) | 1899 | 11007 (536, abc_x4) | 678 | True |
| ../../exp09/ba32_4500_E_incr_prev_step1.L10.rule.json | ba32_4500_10 | 628 | 2197 | 17631 (812, keepxor_abc_x4) | 1963 | 10333 (508, keepxor_abc_x4) | 1260 | True |
| ../../exp09/ba32_4500_E_incr_prev_step1.L5.rule.json | ba32_4500_5 | 445 | 1733 | 13647 (646, keepxor_abc_x4) | 1596 | 7769 (392, abc_x4) | 678 | True |
| ../../exp10/ba32_3000_5__BP0.1/ba32_3000_5_E.rule.json | ba32_3000_5 | 195 | 1117 | 3415 (189, keepxor_abc_x4) | 862 | 2475 (142, abc_x4) | 195 | True |
| ../../exp10/ba32_3000_5__BP1/ba32_3000_5_E.rule.json | ba32_3000_5 | 217 | 1104 | 4275 (232, abc_x4) | 883 | 2415 (139, abc_x4) | 195 | True |
| ../../exp10/ba32_3000_5__conway/ba32_3000_5_E.rule.json | ba32_3000_5 | 266 | 1449 | 8363 (419, abc_x4) | 1062 | 3595 (198, abc_x4) | 195 | True |
| ../../exp10/ba32_3000_5__keep/ba32_3000_5_E.rule.json | ba32_3000_5 | 207 | 715 | 3535 (195, keepxor_abc_x4) | 715 | 2133 (125, abc_x4) | 195 | True |
| ../../exp10/ba32_3000_5__R0/ba32_3000_5_E.rule.json | ba32_3000_5 | 344 | 1722 | 4931 (263, keepxor_abc_x4) | 1066 | 3075 (172, abc_x4) | 195 | True |
| ../../exp10/ba32_3000_5__random/ba32_3000_5_E.rule.json | ba32_3000_5 | 266 | 2346 | 18687 (856, keepxor_abc_x4) | 1233 | 4035 (220, keepxor_abc_x4) | 195 | True |
| ../../exp10/ba32_4500_10__keep/ba32_4500_10_E.rule.json | ba32_4500_10 | 651 | 1940 | 15279 (714, keepxor_abc_x4) | 1940 | 10791 (527, keepxor_abc_x4) | 1260 | True |
| ../../exp10/ba32_4500_10__R0/ba32_4500_10_E.rule.json | ba32_4500_10 | 597 | 2173 | 16551 (767, keepxor_abc_x4) | 1914 | 10983 (535, keepxor_abc_x4) | 1260 | True |
| ../../exp10/ba32_4500_10__random/ba32_4500_10_E.rule.json | ba32_4500_10 | 770 | 2258 | 18807 (861, keepxor_abc_x4) | 2054 | 12375 (593, abc_x4) | 1260 | True |

### Part 4: binarization penalty (selection by hard autonomous rollout; penalty/soft loss are NOT success criteria)

| run | exact seeds | best hard err | hold err | selected it / t | penalty at selection | video sq loss at selection | steps |
|---|---|---|---|---|---|---|---|
| exp12/BP_lam0.1_main (ba32_3000_5) | 0/4 | 0.0070 | 0.0070 | 448 / 40.1 | 48.3 | 15.8 | 9916 |
| exp12/BP_lam1_main (ba32_3000_5) | 0/4 | 0.0070 | 0.0070 | 457 / 40.0 | 23.3 | 19.7 | 9849 |
| exp12/R0_ba32_4500_10_main (ba32_4500_10) | 0/4 | 0.0354 | 0.0616 | 2937 / 240.4 | - | - | 7490 |
| exp07/R0_main (ba32_3000_5) | 0/4 | 0.0057 | 0.0070 | 2619 / 100.1 | - | - | 16854 |
