# Campaign 10 results table (generated from runs/c10/eval/c10_main10_<task>_best.json)

Full span = all autonomous frames (main src 4500-4555, confirm 1800-1859); window = 4540-4555 / 1844-1859 cut from the same rollout.
err = binary pixel error; band = wrong pixels inside the 2-px band of the TRUE contour (fraction of all pixels); lag1/lag4 = tolerant change F1;
mass1 = predicted/true lag-1 changed pixels; blips = spurious 1-frame flips; specks/holes = small (<=3 px) components (auxiliary); passC = C9 window rule;
node = numpy vs Node binary mismatching pixels (max |dv|).

## main: medians over seeds 0-2

| arm | err | band | FP | FN | early/mid/late err | contour F1 | lag1 F1 | lag4 F1 | mass1 | window err | win lag4 F1 | passC seeds | blips | qualifies |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C9 | 0.0129 | 0.0121 | 0.0065 | 0.0065 | 0.0098 / 0.0066 / 0.0226 | 0.977 | 0.726 | 0.910 | 0.62 | 0.0247 | 0.914 | 3/3 | 11 | - |
| A | 0.0068 | 0.0064 | 0.0035 | 0.0033 | 0.0069 / 0.0040 / 0.0096 | 0.986 | 0.861 | 0.949 | 0.71 | 0.0106 | 0.964 | 3/3 | 7 | yes |
| B | 0.0048 | 0.0041 | 0.0026 | 0.0022 | 0.0050 / 0.0025 / 0.0068 | 0.987 | 0.900 | 0.969 | 0.79 | 0.0075 | 0.976 | 3/3 | 10 | yes |
| C | 0.0071 | 0.0067 | 0.0035 | 0.0036 | 0.0070 / 0.0041 / 0.0102 | 0.987 | 0.856 | 0.946 | 0.70 | 0.0111 | 0.960 | 3/3 | 7 | yes |
| BC | 0.0040 | 0.0035 | 0.0023 | 0.0018 | 0.0040 / 0.0020 / 0.0060 | 0.987 | 0.911 | 0.976 | 0.81 | 0.0067 | 0.983 | 3/3 | 13 | yes |

### main: per seed

| model | ckpt step | err | band | lag1 | lag4 | blips | specks pred/true | holes pred/true | window err | node mismatch (max abs dv) | export |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C9_s0 | C9 | 0.0086 | 0.0084 | 0.825 | 0.958 | 15 | 5/0 | 6/19 | 0.0180 | 0 (3.6e-05) | runs/c10/exports/c9_main_C_s0.nca.json |
| C9_s1 | C9 | 0.0197 | 0.0179 | 0.570 | 0.820 | 2 | 0/0 | 2/19 | 0.0414 | 0 (2.5e-05) | runs/c10/exports/c9_main_C_s1.nca.json |
| C9_s2 | C9 | 0.0129 | 0.0121 | 0.726 | 0.910 | 11 | 1/0 | 6/19 | 0.0247 | 0 (1.3e-05) | runs/c10/exports/c9_main_C_s2.nca.json |
| A_s0 | 9988 | 0.0035 | 0.0034 | 0.929 | 0.979 | 7 | 0/0 | 7/19 | 0.0055 | 0 (1.1e-04) | runs/c10/exports/c10_main10_main_best_A_s0.nca.json |
| A_s1 | 10688 | 0.0142 | 0.0128 | 0.675 | 0.878 | 6 | 4/0 | 5/19 | 0.0313 | 0 (5.9e-05) | runs/c10/exports/c10_main10_main_best_A_s1.nca.json |
| A_s2 | 9631 | 0.0068 | 0.0064 | 0.861 | 0.949 | 10 | 2/0 | 23/19 | 0.0106 | 0 (8.5e-05) | runs/c10/exports/c10_main10_main_best_A_s2.nca.json |
| B_s0 | 10688 | 0.0021 | 0.0019 | 0.956 | 0.989 | 6 | 2/0 | 16/19 | 0.0033 | 0 (1.3e-04) | runs/c10/exports/c10_main10_main_best_B_s0.nca.json |
| B_s1 | 7847 | 0.0122 | 0.0104 | 0.748 | 0.919 | 10 | 20/0 | 14/19 | 0.0260 | 0 (9.9e-05) | runs/c10/exports/c10_main10_main_best_B_s1.nca.json |
| B_s2 | 10688 | 0.0048 | 0.0041 | 0.900 | 0.969 | 19 | 11/0 | 10/19 | 0.0075 | 0 (9.5e-05) | runs/c10/exports/c10_main10_main_best_B_s2.nca.json |
| C_s0 | 9631 | 0.0034 | 0.0033 | 0.933 | 0.979 | 12 | 0/0 | 8/19 | 0.0052 | 0 (8.7e-05) | runs/c10/exports/c10_main10_main_best_C_s0.nca.json |
| C_s1 | 10688 | 0.0140 | 0.0124 | 0.688 | 0.881 | 3 | 0/0 | 10/19 | 0.0305 | 0 (1.3e-04) | runs/c10/exports/c10_main10_main_best_C_s1.nca.json |
| C_s2 | 9631 | 0.0071 | 0.0067 | 0.856 | 0.946 | 7 | 3/0 | 21/19 | 0.0111 | 0 (9.1e-05) | runs/c10/exports/c10_main10_main_best_C_s2.nca.json |
| BC_s0 | 19790 | 0.0016 | 0.0014 | 0.964 | 0.989 | 5 | 2/0 | 21/19 | 0.0025 | 0 (1.7e-04) | runs/c10/exports/c10_combo10_main_best_BC_s0.nca.json |
| BC_s1 | 20497 | 0.0110 | 0.0090 | 0.777 | 0.924 | 13 | 33/0 | 17/19 | 0.0245 | 1 (9.5e-05) | runs/c10/exports/c10_combo10_main_best_BC_s1.nca.json |
| BC_s2 | 21192 | 0.0040 | 0.0035 | 0.911 | 0.976 | 21 | 7/0 | 17/19 | 0.0067 | 0 (1.1e-04) | runs/c10/exports/c10_combo10_main_best_BC_s2.nca.json |

## confirm: medians over seeds 0-2

| arm | err | band | FP | FN | early/mid/late err | contour F1 | lag1 F1 | lag4 F1 | mass1 | window err | win lag4 F1 | passC seeds | blips | qualifies |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C9 | 0.0266 | 0.0261 | 0.0092 | 0.0177 | 0.0221 / 0.0282 / 0.0297 | 0.970 | 0.579 | 0.870 | 0.51 | 0.0313 | 0.949 | 3/3 | 37 | - |
| A | 0.0174 | 0.0170 | 0.0056 | 0.0119 | 0.0159 / 0.0227 / 0.0134 | 0.976 | 0.686 | 0.901 | 0.58 | 0.0128 | 0.976 | 3/3 | 30 | yes |
| B | 0.0143 | 0.0135 | 0.0047 | 0.0102 | 0.0130 / 0.0211 / 0.0107 | 0.979 | 0.749 | 0.923 | 0.62 | 0.0099 | 0.987 | 3/3 | 35 | yes |
| C | 0.0172 | 0.0168 | 0.0054 | 0.0117 | 0.0157 / 0.0229 / 0.0127 | 0.976 | 0.693 | 0.905 | 0.58 | 0.0123 | 0.980 | 3/3 | 34 | yes |
| BC | 0.0134 | 0.0127 | 0.0043 | 0.0093 | 0.0118 / 0.0200 / 0.0100 | 0.980 | 0.771 | 0.928 | 0.63 | 0.0091 | 0.986 | 3/3 | 29 | yes |

### confirm: per seed

| model | ckpt step | err | band | lag1 | lag4 | blips | specks pred/true | holes pred/true | window err | node mismatch (max abs dv) | export |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C9_s0 | C9 | 0.0266 | 0.0261 | 0.579 | 0.870 | 37 | 88/104 | 35/30 | 0.0306 | 0 (5.4e-05) | runs/c10/exports/c9_confirm_C_s0.nca.json |
| C9_s1 | C9 | 0.0255 | 0.0249 | 0.596 | 0.873 | 48 | 83/104 | 33/30 | 0.0313 | 0 (7.2e-05) | runs/c10/exports/c9_confirm_C_s1.nca.json |
| C9_s2 | C9 | 0.0286 | 0.0281 | 0.544 | 0.847 | 27 | 47/104 | 39/30 | 0.0336 | 0 (3.2e-05) | runs/c10/exports/c9_confirm_C_s2.nca.json |
| A_s0 | 9970 | 0.0174 | 0.0170 | 0.686 | 0.901 | 42 | 80/104 | 34/30 | 0.0128 | 0 (1.0e-04) | runs/c10/exports/c10_main10_confirm_best_A_s0.nca.json |
| A_s1 | 10297 | 0.0158 | 0.0154 | 0.715 | 0.912 | 24 | 90/104 | 34/30 | 0.0118 | 0 (1.1e-04) | runs/c10/exports/c10_main10_confirm_best_A_s1.nca.json |
| A_s2 | 9970 | 0.0194 | 0.0190 | 0.661 | 0.886 | 30 | 67/104 | 41/30 | 0.0158 | 0 (7.2e-05) | runs/c10/exports/c10_main10_confirm_best_A_s2.nca.json |
| B_s0 | 10297 | 0.0143 | 0.0135 | 0.749 | 0.923 | 43 | 99/104 | 42/30 | 0.0088 | 0 (8.8e-05) | runs/c10/exports/c10_main10_confirm_best_B_s0.nca.json |
| B_s1 | 7907 | 0.0139 | 0.0132 | 0.760 | 0.926 | 35 | 119/104 | 32/30 | 0.0099 | 0 (1.7e-04) | runs/c10/exports/c10_main10_confirm_best_B_s1.nca.json |
| B_s2 | 9970 | 0.0160 | 0.0154 | 0.699 | 0.904 | 35 | 90/104 | 36/30 | 0.0119 | 0 (7.7e-05) | runs/c10/exports/c10_main10_confirm_best_B_s2.nca.json |
| C_s0 | 9626 | 0.0172 | 0.0168 | 0.693 | 0.905 | 36 | 72/104 | 34/30 | 0.0123 | 0 (7.7e-05) | runs/c10/exports/c10_main10_confirm_best_C_s0.nca.json |
| C_s1 | 10297 | 0.0156 | 0.0152 | 0.723 | 0.912 | 31 | 83/104 | 33/30 | 0.0118 | 0 (9.1e-05) | runs/c10/exports/c10_main10_confirm_best_C_s1.nca.json |
| C_s2 | 10297 | 0.0191 | 0.0187 | 0.656 | 0.886 | 34 | 69/104 | 38/30 | 0.0152 | 0 (7.5e-05) | runs/c10/exports/c10_main10_confirm_best_C_s2.nca.json |
| BC_s0 | 19523 | 0.0134 | 0.0127 | 0.771 | 0.928 | 47 | 106/104 | 45/30 | 0.0077 | 0 (1.2e-04) | runs/c10/exports/c10_combo10_confirm_best_BC_s0.nca.json |
| BC_s1 | 20182 | 0.0130 | 0.0124 | 0.774 | 0.929 | 29 | 109/104 | 25/30 | 0.0091 | 0 (2.5e-04) | runs/c10/exports/c10_combo10_confirm_best_BC_s1.nca.json |
| BC_s2 | 20182 | 0.0147 | 0.0141 | 0.727 | 0.914 | 25 | 94/104 | 36/30 | 0.0099 | 0 (6.3e-05) | runs/c10/exports/c10_combo10_confirm_best_BC_s2.nca.json |

