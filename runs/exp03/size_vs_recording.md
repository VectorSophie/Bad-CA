| run | changed pixels | change-list bits* | visited LUT entries x outputs (bits) | SOP literals | SOP bits** |
|---|---|---|---|---|---|
| ba32_3000_5_E | 13 | 170 | 990 | 679 | 3395 |
| ba32_3000_5_E_keep | 13 | 170 | 628 | 453 | 2265 |
| ba32_4500_5_E | 74 | 780 | 1088 | 890 | 4450 |
| ba32_4500_5_E_keep | 74 | 780 | 1250 | 1185 | 5925 |
| bar12_X6 | 140 | 1208 | 1478 | 1792 | 8960 |
| bar12_X6_keep | 140 | 1208 | 1638 | 1969 | 9845 |
| disc12_E | 136 | 1136 | 1144 | 1180 | 5900 |
| disc12_E_keep | 136 | 1136 | 792 | 744 | 3720 |
| disc12_X1 | 136 | 1136 | 1260 | 836 | 4180 |
| disc12_X1_keep | 136 | 1136 | 1548 | 991 | 4955 |
| disc12_X2 | 136 | 1136 | 1080 | 1122 | 5610 |
| disc12_X2_keep | 136 | 1136 | 1046 | 1030 | 5150 |
| disc12_X4 | 136 | 1136 | 980 | - | - |
| disc12_X4_keep | 136 | 1136 | 996 | - | - |

*change list: per changed pixel ceil(log2 cells) bits + per-frame count. **literal = ceil(log2(2n)) bits; term separators ignored (favours SOP).
