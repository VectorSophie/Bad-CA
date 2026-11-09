
### control

| method | target | S | complexity | file B (best baseline B) | exact | mean err | hold | err/hold | change-IoU | tol-space | tol-time | horizon/T | search cost | Node replay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2 | life_b3s23 | 1 | B3/S23 (3 set bits of 18) | 141 (710 changelist) | True | 0.0 | 0.4066 | 0.0 | 1.0 | 0.0 | 0.0 | 12/12 | 10.3 CPU-s | True |
| C4 | pcA | 1 | 2 active gates (of G=8), seed 0 | 27 (116 lzma) | True | 0.0 | 0.4847 | 0.0 | 1.0 | 0.0 | 0.0 | 6/6 | 43.7 CPU-s | True |
| C4 | pcA | 1 | 4 active gates (of G=8), seed 1 | 30 (116 lzma) | True | 0.0 | 0.4847 | 0.0 | 1.0 | 0.0 | 0.0 | 6/6 | 11.3 CPU-s | True |
| C4 | pcA | 1 | 2 active gates (of G=8), seed 2 | 27 (116 lzma) | True | 0.0 | 0.4847 | 0.0 | 1.0 | 0.0 | 0.0 | 6/6 | 2.7 CPU-s | True |
| C4 | pcA | 1 | 2 active gates (of G=8), seed 3 | 27 (116 lzma) | True | 0.0 | 0.4847 | 0.0 | 1.0 | 0.0 | 0.0 | 6/6 | 7.3 CPU-s | True |
| C5 | pcA_long | 1 | 8 live gates, seed 1000 of 4 | 37 (306 zlib) | True | 0.0 | 0.4608 | 0.0 | 1.0 | 0.0 | 0.0 | 18/18 | 31.0 GPU-s train + 5.1 s compile | True |
| C4 | pcB | 2 | 3 active gates (of G=8), seed 2 | 29 (90 zlib) | True | 0.0 | 0.5444 | 0.0 | 1.0 | 0.0 | 0.0 | 6/6 | 175.9 CPU-s | True |
| C4 | pcB | 2 | 3 active gates (of G=8), seed 3 | 29 (90 zlib) | True | 0.0 | 0.5444 | 0.0 | 1.0 | 0.0 | 0.0 | 6/6 | 128.6 CPU-s | True |
| C4 | pcB | 2 | 5 active gates (of G=8), seed 1 | 32 (90 zlib) | False | 0.0972 | 0.5444 | 0.179 | 0.41 | 0.0 | 0.0667 | 6/6 | 300.0 CPU-s | True |
| C4 | pcB | 2 | 1 active gates (of G=8), seed 0 | 25 (90 zlib) | False | 0.1056 | 0.5444 | 0.194 | 0.352 | 0.0 | 0.0694 | 6/6 | 300.0 CPU-s | True |
| C3 | so_ctrl | 1 | tree: 3 conditions, 4 leaves (3 gates compiled) | 44 (311 zlib) | True | 0.0 | 0.4462 | 0.0 | 1.0 | 0.0 | 0.0 | 10/10 | 0.1 CPU-s | True |

### common

| method | target | S | complexity | file B (best baseline B) | exact | mean err | hold | err/hold | change-IoU | tol-space | tol-time | horizon/T | search cost | Node replay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2 | ba32_3000_5 | 1 | B/S2345678 (7 set bits of 18) | 66 (80 changelist) | False | 0.0088 | 0.0088 | 1.0 | 0.0 | 0.0 | 0.0046 | 5/5 | 5.6 CPU-s | True |
| C2 | ba32_3000_5 | 2 | B012346/S (6 set bits of 18) | 127 (80 changelist) | False | 0.0088 | 0.0088 | 1.0 | 0.0 | 0.0 | 0.0052 | 1/5 | 9.6 CPU-s | True |
| C3 | ba32_3000_5 | 1 | tree: 3 conditions, 4 leaves (2 gates compiled) | 62 (80 changelist) | False | 0.0072 | 0.0088 | 0.815 | 0.143 | 0.0 | 0.0049 | 5/5 | 0.4 CPU-s | True |
| C4 | ba32_3000_5 | 2 | 4 active gates (of G=16), seed 0 | 65 (80 changelist) | False | 0.0078 | 0.0088 | 0.889 | 0.0 | 0.0 | 0.0049 | 1/5 | 300.0 CPU-s | True |
| C4 | ba32_3000_5 | 2 | 3 active gates (of G=16), seed 3 | 63 (80 changelist) | False | 0.0081 | 0.0088 | 0.926 | 0.0 | 0.0 | 0.0052 | 1/5 | 300.0 CPU-s | True |
| C4 | ba32_3000_5 | 2 | 0 active gates (of G=16), seed 1 | 58 (80 changelist) | False | 0.0088 | 0.0088 | 1.0 | 0.0 | 0.0 | 0.0046 | 5/5 | 300.0 CPU-s | True |
| C4 | ba32_3000_5 | 2 | 0 active gates (of G=16), seed 2 | 58 (80 changelist) | False | 0.0088 | 0.0088 | 1.0 | 0.0 | 0.0 | 0.0046 | 5/5 | 300.0 CPU-s | True |
| C2 | ba48_4500_60 | 2 | B0134/S0124 (8 set bits of 18) | 151 (1675 lzma) | False | 0.1465 | 0.1852 | 0.791 | 0.025 | 0.085 | 0.1427 | 1/60 | 271.3 CPU-s | True |
| C2 | ba48_4500_60 | 1 | B4/S345678 (7 set bits of 18) | 125 (1675 lzma) | False | 0.1658 | 0.1852 | 0.895 | 0.007 | 0.1034 | 0.1611 | 1/60 | 142.5 CPU-s | True |
| C3 | ba48_4500_60 | 1 | tree: 1466 conditions, 1467 leaves (2267 gates compiled) | 7032 (1675 lzma) | False | 0.1247 | 0.1852 | 0.673 | 0.036 | 0.0637 | 0.1194 | 49/60 | 6.9 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 6 active gates (of G=32), seed 3 | 89 (1675 lzma) | False | 0.0887 | 0.1852 | 0.479 | 0.017 | 0.0319 | 0.0847 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 8 active gates (of G=16), seed 0 | 92 (1675 lzma) | False | 0.0919 | 0.1852 | 0.496 | 0.02 | 0.0354 | 0.0878 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 10 active gates (of G=16), seed 2 | 96 (1675 lzma) | False | 0.092 | 0.1852 | 0.497 | 0.021 | 0.0354 | 0.0871 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 6 active gates (of G=8), seed 3 | 89 (1675 lzma) | False | 0.0945 | 0.1852 | 0.51 | 0.014 | 0.0373 | 0.0905 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 3 active gates (of G=8), seed 1 | 84 (1675 lzma) | False | 0.097 | 0.1852 | 0.524 | 0.012 | 0.0394 | 0.0928 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 6 active gates (of G=16), seed 1 | 89 (1675 lzma) | False | 0.0999 | 0.1852 | 0.539 | 0.013 | 0.0439 | 0.0962 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 6 active gates (of G=32), seed 2 | 89 (1675 lzma) | False | 0.1004 | 0.1852 | 0.542 | 0.012 | 0.044 | 0.0956 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 8 active gates (of G=16), seed 3 | 92 (1675 lzma) | False | 0.1038 | 0.1852 | 0.56 | 0.044 | 0.0453 | 0.0992 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 7 active gates (of G=32), seed 0 | 91 (1675 lzma) | False | 0.1039 | 0.1852 | 0.561 | 0.009 | 0.0449 | 0.0993 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 4 active gates (of G=32), seed 1 | 86 (1675 lzma) | False | 0.1054 | 0.1852 | 0.569 | 0.03 | 0.0482 | 0.1009 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 4 active gates (of G=8), seed 2 | 86 (1675 lzma) | False | 0.1054 | 0.1852 | 0.569 | 0.03 | 0.0482 | 0.1009 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_4500_60 | 2 | 3 active gates (of G=8), seed 0 | 84 (1675 lzma) | False | 0.1287 | 0.1852 | 0.695 | 0.01 | 0.0681 | 0.1248 | 1/60 | 600.0 CPU-s | True |
| C5 | ba48_4500_60 | 2 | 0 live gates, seed 1001 of 4 | 79 (1675 lzma) | False | 0.1852 | 0.1852 | 1.0 | 0.0 | 0.1224 | 0.1798 | 60/60 | 1502.2 GPU-s train + 8.8 s compile | True |

### diagnostic

| method | target | S | complexity | file B (best baseline B) | exact | mean err | hold | err/hold | change-IoU | tol-space | tol-time | horizon/T | search cost | Node replay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2 | ba48_1800_60 | 1 | B5/S235678 (7 set bits of 18) | 176 (3031 lzma) | False | 0.1378 | 0.1582 | 0.871 | 0.008 | 0.0284 | 0.1307 | 1/60 | 141.4 CPU-s | True |
| C3 | ba48_1800_60 | 1 | tree: 53 conditions, 54 leaves (79 gates compiled) | 283 (3031 lzma) | False | 0.1245 | 0.1582 | 0.787 | 0.006 | 0.0272 | 0.1168 | 1/60 | 5.7 CPU-s | True |
| C4 | ba48_1800_60 | 2 | 8 active gates (of G=16), seed 3 | 132 (3031 lzma) | False | 0.1192 | 0.1582 | 0.754 | 0.003 | 0.023 | 0.111 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_1800_60 | 2 | 6 active gates (of G=16), seed 0 | 128 (3031 lzma) | False | 0.1315 | 0.1582 | 0.831 | 0.004 | 0.03 | 0.1239 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_1800_60 | 2 | 7 active gates (of G=16), seed 2 | 130 (3031 lzma) | False | 0.1318 | 0.1582 | 0.833 | 0.01 | 0.0303 | 0.1259 | 1/60 | 600.0 CPU-s | True |
| C4 | ba48_1800_60 | 2 | 2 active gates (of G=16), seed 1 | 122 (3031 lzma) | False | 0.1363 | 0.1582 | 0.862 | 0.007 | 0.0323 | 0.1295 | 1/60 | 600.0 CPU-s | True |
| C2 | bar12 | 1 | B/S5 (1 set bits of 18) | 61 (112 changelist) | False | 0.1439 | 0.1894 | 0.76 | 0.039 | 0.0 | 0.1263 | 11/12 | 3.7 CPU-s | True |
| C3 | bar12 | 1 | tree: 198 conditions, 199 leaves (285 gates compiled) | 695 (112 changelist) | False | 0.1225 | 0.1894 | 0.647 | 0.141 | 0.0 | 0.1048 | 11/12 | 1.1 CPU-s | True |
| C4 | bar12 | 4 | 8 active gates (of G=16), seed 1 | 24 (112 changelist) | False | 0.1408 | 0.1894 | 0.743 | 0.085 | 0.0006 | 0.1319 | 11/12 | 300.0 CPU-s | True |
| C4 | bar12 | 4 | 7 active gates (of G=16), seed 0 | 23 (112 changelist) | False | 0.1439 | 0.1894 | 0.76 | 0.093 | 0.0006 | 0.1338 | 11/12 | 300.0 CPU-s | True |
| C2 | disc12 | 2 | B3/S345678 (7 set bits of 18) | 58 (55 zlib) | False | 0.1204 | 0.2176 | 0.553 | 0.167 | 0.0139 | 0.0602 | 6/7 | 3.9 CPU-s | True |
| C2 | disc12 | 1 | B3/S34578 (6 set bits of 18) | 66 (55 zlib) | False | 0.1574 | 0.2176 | 0.723 | 0.087 | 0.0324 | 0.0833 | 6/7 | 2.2 CPU-s | True |
| C3 | disc12 | 1 | tree: 2 conditions, 3 leaves (1 gates compiled) | 15 (55 zlib) | False | 0.2083 | 0.2176 | 0.957 | 0.132 | 0.0787 | 0.1574 | 5/7 | 1.2 CPU-s | True |
| C4 | disc12 | 2 | 10 active gates (of G=16), seed 0 | 30 (55 zlib) | False | 0.1215 | 0.2176 | 0.559 | 0.162 | 0.0174 | 0.0613 | 6/7 | 300.0 CPU-s | True |
| C4 | disc12 | 2 | 5 active gates (of G=16), seed 1 | 21 (55 zlib) | False | 0.125 | 0.2176 | 0.574 | 0.147 | 0.0139 | 0.0231 | 6/7 | 300.0 CPU-s | True |
| C2 | translate24 | 1 | B/S58 (2 set bits of 18) | 70 (55 zlib) | False | 0.0568 | 0.0966 | 0.588 | 0.077 | 0.0243 | 0.0549 | 12/12 | 10.3 CPU-s | True |
| C3 | translate24 | 1 | tree: 1 conditions, 2 leaves (0 gates compiled) | 20 (55 zlib) | True | 0.0 | 0.0966 | 0.0 | 1.0 | 0.0 | 0.0 | 12/12 | 0.3 CPU-s | True |
| C4 | translate24 | 1 | 1 active gates (of G=4), seed 0 | 20 (55 zlib) | True | 0.0 | 0.0966 | 0.0 | 1.0 | 0.0 | 0.0 | 12/12 | 0.1 CPU-s | True |
| C4 | translate24 | 1 | 0 active gates (of G=4), seed 1 | 19 (55 zlib) | True | 0.0 | 0.0966 | 0.0 | 1.0 | 0.0 | 0.0 | 12/12 | 0.0 CPU-s | True |

### CPU ledger (core-s; wall-s separately)

- c2_main: CPU 453, wall 454
- c2_smoke: CPU 11, wall 11
- c3_control: CPU 0, wall 0
- c3_main: CPU 24, wall 26
- c4_main: CPU 10585, wall 10592
- c4_smoke: CPU 56, wall 56
- fu_c2: CPU 142, wall 142
- fu_c3: CPU 7, wall 8
- fu_c4: CPU 2404, wall 2404
- **total CPU 3.80 core-h**
