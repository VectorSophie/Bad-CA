### Search (direct circuit synthesis; status per G, CPU s in brackets)

| clip | structure | encoding | per G | largest G proved UNSAT | first SAT G | CPU core-s |
|---|---|---|---|---|---|---|
| ba32_3000_5 | k1_S2 | lex | G3 UNSAT (30), G4 UNSAT (605), G6 UNKNOWN (60), G8 UNKNOWN (60), G8 UNKNOWN (1800), G10 UNKNOWN (60), G12 UNKNOWN (1800) | 4 | - | 4416 |
| ba32_3000_5 | k1_S2 | v1 | G0 UNSAT (0), G1 UNSAT (1), G2 UNSAT (2), G3 UNSAT (25), G4 UNKNOWN (60), G5 UNKNOWN (60), G6 UNKNOWN (60), G7 UNKNOWN (60), G8 UNKNOWN (60), G9 UNKNOWN (60), G10 UNKNOWN (60), G11 UNKNOWN (60), G12 UNKNOWN (60) | 3 | - | 569 |
| ba32_3000_5 | k1_S2 | v3 | G3 UNSAT (55), G4 UNKNOWN (120), G5 UNKNOWN (120), G6 UNKNOWN (120), G16 UNKNOWN (300), G32 UNKNOWN (300) | 3 | - | 1016 |
| ba32_3000_5 | k1_S4 | lex | G1 UNSAT (3), G2 UNSAT (49), G3 UNKNOWN (600), G12 UNKNOWN (1200) | 2 | - | 1852 |
| ba32_3000_5 | k1s_S2 | lex | G1 UNSAT (1), G2 UNSAT (12), G3 UNSAT (166), G12 UNKNOWN (1800) | 3 | - | 1980 |
| ba32_3000_5 | k2_S2 | lex | G1 UNSAT (1), G2 UNSAT (4), G3 UNSAT (47), G12 UNKNOWN (1200) | 3 | - | 1251 |
| ba32_4500_10 | k1_S2 | lex | G1 UNSAT (2), G2 UNSAT (10) | 2 | - | 12 |
| ba32_4500_5 | k1_S2 | lex | G1 UNSAT (1), G2 UNSAT (4), G3 UNSAT (40), G12 UNKNOWN (1200) | 3 | - | 1245 |
| bar12 | k1_S4 | lex | G2 UNSAT (18), G3 UNSAT (542), G12 UNKNOWN (1200) | 3 | - | 1760 |
| bar12 | k1_S4 | v1 | G0 UNSAT (0), G1 UNSAT (1), G2 UNSAT (19), G3 UNKNOWN (60), G4 UNKNOWN (60), G5 UNKNOWN (60), G6 UNKNOWN (60), G7 UNKNOWN (60), G8 UNKNOWN (60), G9 UNKNOWN (60), G10 UNKNOWN (60) | 2 | - | 501 |
| bar12 | k1_S4 | v3 | G2 UNSAT (43), G3 UNKNOWN (120), G4 UNKNOWN (120), G5 UNKNOWN (120), G6 UNKNOWN (120) | 2 | - | 525 |
| disc12 | k1_S2 | lex | G3 UNSAT (5), G4 UNSAT (144), G5 UNKNOWN (1800), G12 UNKNOWN (1200) | 4 | - | 3149 |
| disc12 | k1_S2 | v1 | G0 UNSAT (0), G1 UNSAT (0), G2 UNSAT (0), G3 UNSAT (5), G4 UNKNOWN (60), G4 UNSAT (234), G5 UNKNOWN (60), G6 UNKNOWN (60), G7 UNKNOWN (60), G8 UNKNOWN (60), G9 UNKNOWN (60), G10 UNKNOWN (60), G16 UNKNOWN (300) | 4 | - | 960 |
| disc12 | k1_S2 | v3 | G3 UNSAT (27), G4 UNKNOWN (120), G5 UNKNOWN (120), G6 UNKNOWN (120), G7 UNKNOWN (120), G8 UNKNOWN (120), G16 UNKNOWN (300), G32 UNKNOWN (300) | 3 | - | 1228 |
| disc12 | k1s_S1 | lex | G1 UNSAT (0), G2 UNSAT (0), G3 UNSAT (0) | 3 | - | 1 |
| disc12:p4 | k1_S2 | lex | G0 UNSAT (0), G1 UNSAT (0), G2 UNSAT (0), G3 UNSAT (3), G4 UNSAT (82) | 4 | - | 86 |
| pcA | k1_S1 | v1 | G0 UNSAT (0), G1 UNSAT (0), G2 SAT (0) | 1 | 2 | 0 |
| pcA:p3 | k1_S1 | lex | G0 UNSAT (0), G1 UNSAT (0), G2 SAT (0) | 1 | 2 | 0 |
| pcB | k1_S2 | lex | G2 UNSAT (0) | 2 | - | 0 |
| pcB | k1_S2 | v1 | G0 UNSAT (0), G1 UNSAT (0), G2 UNSAT (1), G2 UNSAT (1), G3 SAT (2), G3 SAT (1) | 2 | 3 | 5 |
| pcB | k1_S2 | v3 | G2 UNSAT (8), G3 SAT (16) | 2 | 3 | 24 |
| pcB:p3 | k1_S2 | lex | G0 UNSAT (0), G1 UNSAT (0), G2 UNSAT (0), G3 SAT (1) | 2 | 3 | 2 |
| translate24 | k0_S1 | v1 | G0 SAT (0) | - | 0 | 0 |
| translate24:p3 | k0_S1 | lex | G0 SAT (0) | - | 0 | 0 |

### Gate-count bounds per clip x structure (all encodings combined; v1/lex/v3 have the same solution existence)

| clip | structure | proved: no circuit with <= G gates | smallest SAT G found | minimal? | largest UNKNOWN G probed |
|---|---|---|---|---|---|
| ba32_3000_5 | k1_S2 | 4 | - | no | 32 |
| ba32_3000_5 | k1_S4 | 2 | - | no | 12 |
| ba32_3000_5 | k1s_S2 | 3 | - | no | 12 |
| ba32_3000_5 | k2_S2 | 3 | - | no | 12 |
| ba32_4500_10 | k1_S2 | 2 | - | no | - |
| ba32_4500_5 | k1_S2 | 3 | - | no | 12 |
| bar12 | k1_S4 | 3 | - | no | 12 |
| disc12 | k1_S2 | 4 | - | no | 32 |
| disc12 | k1s_S1 | 3 | - | no | - |
| disc12:p4 | k1_S2 | 4 | - | no | - |
| pcA | k1_S1 | 1 | 2 | yes | - |
| pcA:p3 | k1_S1 | 1 | 2 | yes | - |
| pcB | k1_S2 | 2 | 3 | yes | - |
| pcB:p3 | k1_S2 | 2 | 3 | yes | - |
| translate24 | k0_S1 | - | 0 | yes | - |
| translate24:p3 | k0_S1 | - | 0 | yes | - |

### Total file bytes per clip (every file decoded from its bytes; CA files also by the Node player)

| clip | T×H×W | changelist | xor_rle | frame_rle | zlib | bz2 | lzma | best LUT-route file (bytes, flow) | best LUT-route circuit (bytes, gates) | best direct-synthesis circuit (bytes, G) | CA beats every tested baseline? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pcA | 6×12×12 | 129 | 177 | 176 | 117 | 191 | 116 | - | - | 27 (G=2, k1_S1) | **yes** |
| pcB | 6×12×12 | 96 | 124 | 92 | 90 | 148 | 92 | - | - | 29 (G=3, k1_S2) | **yes** |
| translate24 | 12×24×24 | 151 | 212 | 162 | 55 | 104 | 59 | - | - | 19 (G=0, k0_S1) | **yes** |
| disc12 | 7×12×12 | 81 | 110 | 80 | 55 | 101 | 65 | 154 (lut, full) | 632 (265) | - | no |
| bar12 | 12×12×12 | 112 | 157 | 145 | 136 | 152 | 141 | 272 (lut, full) | 1768 (664) | - | no |
| ba32_3000_5 | 5×24×32 | 80 | 88 | 261 | 80 | 115 | 83 | 129 (lut, full) | 322 (125) | - | no |
| ba32_4500_5 | 5×24×32 | 130 | 156 | 205 | 155 | 179 | 141 | 191 (lut, full) | 753 (296) | - | no |
| ba32_4500_10 | 10×24×32 | 203 | 251 | 395 | 217 | 234 | 202 | 278 (lut, full) | 1308 (498) | - | no |

### Decode cost (CA files): updates and time

| clip | file | bytes | updates (cells × generations) | Python decode s | Node decode ms |
|---|---|---|---|---|---|
| pcA | pcA__k1_S1__G2 | 27 | 720 | 0.001 | 2.01 |
| pcB | pcB__k1_S2__G3 | 29 | 1440 | 0.002 | 3.48 |
| pcB | pcB__k1_S2__G3_ent | 29 | 1440 | 0.002 | 3.51 |
| translate24 | translate24__k0_S1__G0 | 19 | 6336 | 0.002 | 2.25 |
| disc12 | exp03c__disc12_E.full.lut | 154 | 1728 | 0.002 | 2.6 |
| disc12 | exp03c__disc12_E.reach.lut | 154 | 1728 | 0.002 | 5.8 |
| bar12 | exp03c__bar12_X6.full.lut | 272 | 6336 | 0.007 | 6.76 |
| bar12 | exp03c__bar12_X6.reach.lut | 272 | 6336 | 0.005 | 6.91 |
| ba32_3000_5 | exp03c__ba32_3000_5_E.full.lut | 129 | 6144 | 0.002 | 3.93 |
| ba32_3000_5 | exp03c__ba32_3000_5_E.reach.lut | 129 | 6144 | 0.002 | 3.66 |
| ba32_4500_5 | exp03c__ba32_4500_5_E.full.lut | 191 | 6144 | 0.002 | 3.7 |
| ba32_4500_5 | exp03c__ba32_4500_5_E.reach.lut | 191 | 6144 | 0.002 | 4.83 |
| ba32_4500_10 | exp09__ba32_4500_E_incr_keep_step1.L10.full.lut | 278 | 13824 | 0.003 | 5.66 |
| ba32_4500_10 | exp09__ba32_4500_E_incr_keep_step1.L10.reach.lut | 278 | 13824 | 0.003 | 5.78 |

### CPU ledger (core-s, user+sys incl. children)

- certs: 7
- files_verify: 66
- impl_test: 66
- pilot_encode_probe: 47
- prefix_diag: 5
- search:c5/bottleneck: 3094
- search:c5/lex: 433
- search:c5/main: 15375
- search:c5/pilot: 1679
- summarize: 6
- **total: 20779 core-s = 5.77 core-h**; GPU: 0
