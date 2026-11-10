# Campaign 11 results (every seed; full-interval autonomous replay from the export, original time alignment)

err = binary pixel error; early/mid/late = error per third; lag1 = tolerant change F1; mass = predicted/true changed pixels (lag 1); thinFG / thinBG = recall of the original's thin parts / thin gaps; specks = small (<=3 px) fg components pred/true; blips = spurious 1-frame flips; node = numpy vs independent Node binary mismatching pixels; sust = sustained-motion gate (long intervals).

| task | model | export | err | early / mid / late | lag1 | mass | thinFG | thinBG | specks | blips | C9 rule / sust | node px | bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | C10 B+C (10,320) ref | c10_combo10_main_best_BC_s0.nca.json | 0.0016 | 0.0017 / 0.0008 / 0.0023 | 0.964 | 0.90 | 0.976 | 0.901 | 2/0 | 5 | True | - | 56234 |
| main | C10 B+C (10,320) ref | c10_combo10_main_best_BC_s1.nca.json | 0.0110 | 0.0058 / 0.0050 / 0.0223 | 0.777 | 0.61 | 0.787 | 0.673 | 33/0 | 13 | True | - | 56233 |
| main | C10 B+C (10,320) ref | c10_combo10_main_best_BC_s2.nca.json | 0.0040 | 0.0041 / 0.0020 / 0.0060 | 0.911 | 0.81 | 0.928 | 0.816 | 7/0 | 21 | True | - | 56232 |
| main | C9 1,304 distilled (A, historical) | nca_main_C_small8x16_s0.nca.json | 0.0226 | 0.0215 / 0.0132 / 0.0332 | 0.576 | 0.56 | 0.553 | 0.364 | 2/0 | 4 | True | - | 8208 |
| main | C9 1,304 distilled (A, historical) | nca_main_C_small8x16_s1.nca.json | 0.0269 | 0.0211 / 0.0158 / 0.0442 | 0.472 | 0.47 | 0.528 | 0.387 | 2/0 | 11 | True | - | 8208 |
| main | C9 1,304 distilled (A, historical) | nca_main_C_small8x16_s2.nca.json | 0.0253 | 0.0240 / 0.0138 / 0.0380 | 0.535 | 0.52 | 0.468 | 0.412 | 5/0 | 8 | True | - | 8208 |
| main | C11 1,304 direct (B) | p1_main_8x16_D_D_s0.ncab | 0.0178 | 0.0194 / 0.0091 / 0.0248 | 0.677 | 0.65 | 0.688 | 0.569 | 50/0 | 32 | True | 0 | 5321 |
| main | C11 1,304 direct (B) | p1_main_8x16_D_D_s1.ncab | 0.0174 | 0.0188 / 0.0084 / 0.0249 | 0.684 | 0.65 | 0.700 | 0.585 | 29/0 | 23 | True | 0 | 5322 |
| main | C11 1,304 direct (B) | p1_main_8x16_D_D_s2.ncab | 0.0175 | 0.0182 / 0.0075 / 0.0267 | 0.642 | 0.62 | 0.704 | 0.541 | 32/0 | 17 | True | 0 | 5334 |
| main | C11 1,304 distilled (C) | p1_main_8x16_T_T_s0.ncab | 0.0213 | 0.0177 / 0.0115 / 0.0348 | 0.599 | 0.55 | 0.618 | 0.502 | 23/0 | 11 | True | 0 | 5406 |
| main | C11 1,304 distilled (C) | p1_main_8x16_T_T_s1.ncab | 0.0218 | 0.0187 / 0.0120 / 0.0349 | 0.588 | 0.54 | 0.643 | 0.521 | 41/0 | 12 | True | 0 | 5403 |
| main | C11 1,304 distilled (C) | p1_main_8x16_T_T_s2.ncab | 0.0225 | 0.0200 / 0.0107 / 0.0370 | 0.557 | 0.54 | 0.661 | 0.535 | 42/0 | 7 | True | 0 | 5401 |
| main | C11 3,884 direct | p1_main_12x32_D_D_s0.ncab | 0.0062 | 0.0038 / 0.0020 / 0.0129 | 0.876 | 0.79 | 0.894 | 0.802 | 13/0 | 16 | True | 0 | 14944 |
| main | C11 3,884 direct | p1_main_12x32_D_D_s1.ncab | 0.0058 | 0.0049 / 0.0028 / 0.0098 | 0.876 | 0.81 | 0.908 | 0.802 | 13/0 | 22 | True | 0 | 14917 |
| main | C11 3,884 direct | p1_main_12x32_D_D_s2.ncab | 0.0079 | 0.0069 / 0.0037 / 0.0131 | 0.852 | 0.74 | 0.853 | 0.747 | 17/0 | 22 | True | 0 | 14917 |
| confirm | C10 B+C (10,320) ref | c10_combo10_confirm_best_BC_s0.nca.json | 0.0134 | 0.0118 / 0.0196 / 0.0084 | 0.771 | 0.64 | 0.879 | 0.913 | 106/104 | 47 | True | - | 56242 |
| confirm | C10 B+C (10,320) ref | c10_combo10_confirm_best_BC_s1.nca.json | 0.0130 | 0.0089 / 0.0201 / 0.0101 | 0.774 | 0.63 | 0.880 | 0.920 | 109/104 | 29 | True | - | 56240 |
| confirm | C10 B+C (10,320) ref | c10_combo10_confirm_best_BC_s2.nca.json | 0.0147 | 0.0127 / 0.0209 / 0.0104 | 0.727 | 0.59 | 0.860 | 0.919 | 94/104 | 25 | True | - | 56242 |
| confirm | C9 1,304 distilled (A, historical) | nca_confirm_C_small8x16_s0.nca.json | 0.0591 | 0.0538 / 0.0504 / 0.0739 | 0.257 | 0.25 | 0.525 | 0.716 | 84/104 | 6 | False | - | 8219 |
| confirm | C9 1,304 distilled (A, historical) | nca_confirm_C_small8x16_s1.nca.json | 0.0569 | 0.0577 / 0.0500 / 0.0632 | 0.313 | 0.30 | 0.518 | 0.656 | 98/104 | 16 | True | - | 8219 |
| confirm | C9 1,304 distilled (A, historical) | nca_confirm_C_small8x16_s2.nca.json | 0.0600 | 0.0594 / 0.0507 / 0.0703 | 0.285 | 0.30 | 0.525 | 0.650 | 130/104 | 29 | True | - | 8219 |
| confirm | C11 1,304 direct (B) | p1_confirm_8x16_D_D_s0.ncab | 0.0587 | 0.0605 / 0.0449 / 0.0713 | 0.285 | 0.32 | 0.590 | 0.696 | 223/104 | 25 | False | 0 | 5383 |
| confirm | C11 1,304 direct (B) | p1_confirm_8x16_D_D_s1.ncab | 0.0470 | 0.0593 / 0.0413 / 0.0400 | 0.394 | 0.40 | 0.672 | 0.698 | 150/104 | 17 | True | 0 | 5375 |
| confirm | C11 1,304 direct (B) | p1_confirm_8x16_D_D_s2.ncab | 0.0431 | 0.0528 / 0.0357 / 0.0406 | 0.422 | 0.45 | 0.674 | 0.721 | 143/104 | 39 | True | 1 | 5378 |
| confirm | C11 3,884 direct | p1_confirm_12x32_D_D_s0.ncab | 0.0315 | 0.0291 / 0.0285 / 0.0372 | 0.501 | 0.45 | 0.741 | 0.801 | 168/104 | 20 | True | 0 | 14969 |
| confirm | C11 3,884 direct | p1_confirm_12x32_D_D_s1.ncab | 0.0287 | 0.0321 / 0.0279 / 0.0259 | 0.567 | 0.52 | 0.782 | 0.833 | 159/104 | 34 | True | 0 | 14956 |
| confirm | C11 3,884 direct | p1_confirm_12x32_D_D_s2.ncab | 0.0252 | 0.0281 / 0.0247 / 0.0228 | 0.634 | 0.57 | 0.814 | 0.831 | 197/104 | 44 | True | 0 | 14975 |
| L120 | C11 10,320 direct | L120_big16x64_r2_s0.ncab | 0.0304 | 0.0415 / 0.0321 / 0.0174 | 0.597 | 0.53 | 0.787 | 0.784 | 253/169 | 32 | True | 1 | 38977 |
| L120 | C11 10,320 direct | L120_big16x64_r2_s1.ncab | 0.0323 | 0.0416 / 0.0361 / 0.0189 | 0.575 | 0.52 | 0.781 | 0.774 | 275/169 | 29 | True | 0 | 38945 |
| L120 | C11 3,884 direct | L120_small12x32_s0.ncab | 0.0611 | 0.0751 / 0.0576 / 0.0504 | 0.319 | 0.35 | 0.584 | 0.636 | 342/169 | 8 | False | 0 | 14990 |
| L120 | C11 3,884 direct | L120_small12x32_s1.ncab | 0.0537 | 0.0740 / 0.0496 / 0.0371 | 0.389 | 0.40 | 0.629 | 0.655 | 235/169 | 12 | False | 1 | 14986 |
| L240 | C11 10,320 direct (1 seed, exploratory) | L240_big16x64_scratch_s0.ncab | 0.0656 | 0.0686 / 0.1000 / 0.0276 | 0.354 | 0.34 | 0.556 | 0.617 | 696/292 | 13 | False | 2 | 38973 |

## Medians

| task | model | seeds | err | late | lag1 | mass | thinFG | thinBG |
|---|---|---|---|---|---|---|---|---|
| main | C10 B+C (10,320) ref | 3 | 0.0040 | 0.0060 | 0.911 | 0.81 | 0.928 | 0.816 |
| main | C9 1,304 distilled (A, historical) | 3 | 0.0253 | 0.0380 | 0.535 | 0.52 | 0.528 | 0.387 |
| main | C11 1,304 direct (B) | 3 | 0.0175 | 0.0249 | 0.677 | 0.65 | 0.700 | 0.569 |
| main | C11 1,304 distilled (C) | 3 | 0.0218 | 0.0349 | 0.588 | 0.54 | 0.643 | 0.521 |
| main | C11 3,884 direct | 3 | 0.0062 | 0.0129 | 0.876 | 0.79 | 0.894 | 0.802 |
| confirm | C10 B+C (10,320) ref | 3 | 0.0134 | 0.0101 | 0.771 | 0.63 | 0.879 | 0.919 |
| confirm | C9 1,304 distilled (A, historical) | 3 | 0.0591 | 0.0703 | 0.285 | 0.30 | 0.525 | 0.656 |
| confirm | C11 1,304 direct (B) | 3 | 0.0470 | 0.0406 | 0.394 | 0.40 | 0.672 | 0.698 |
| confirm | C11 3,884 direct | 3 | 0.0287 | 0.0259 | 0.567 | 0.52 | 0.782 | 0.831 |
| L120 | C11 10,320 direct | 2 | 0.0314 | 0.0181 | 0.586 | 0.52 | 0.784 | 0.779 |
| L120 | C11 3,884 direct | 2 | 0.0574 | 0.0437 | 0.354 | 0.38 | 0.607 | 0.646 |
| L240 | C11 10,320 direct (1 seed, exploratory) | 1 | 0.0656 | 0.0276 | 0.354 | 0.34 | 0.556 | 0.617 |
