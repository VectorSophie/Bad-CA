# Research timeline (C1–C13)

**What this page is:** a short history of how the project arrived at the current demo.
- **Each entry:** the method, the key finding, the limit, and the record it comes from.
- **Full records** are in `docs/` (plans `CAMPAIGN_NN.md`, results `CHECKPOINT_NN.md`); the per-file index is
  `docs/research/README.md`.
- **Numbers** are copied from those records.

| # | method | key finding | limit | record |
|---|---|---|---|---|
| C1 | Exact lookup-table (LUT) rules and an independent replayer | Exact replay of short clips is possible with large LUTs. A uniform first frame needs care. | LUTs are far larger than recordings. | [CHECKPOINT_01](../CHECKPOINT_01.md) |
| C2 | State-representation ablation (SAT, DiffLogic) | Hidden bits help SAT find exact rules on short clips. Soft rules do not harden well. | Clips of a few frames only. | [CHECKPOINT_02](../CHECKPOINT_02.md) |
| C3 | Length scaling of exact SAT rules | Exact rules up to 10 frames. | 20–40 frames time out; rules exceed every recording baseline. | [CHECKPOINT_03](../CHECKPOINT_03.md) |
| C4 | SAT-centred search (incremental, learned starts) | No speed-up over cold SAT at equal budget. | Compression still not shown. | [CHECKPOINT_04](../CHECKPOINT_04.md) |
| C5 | Direct SAT synthesis of small circuits, compared by total file bytes | Small circuits win only on synthetic positive controls. | No compression on Bad Apple. | [CHECKPOINT_05](../CHECKPOINT_05.md) |
| C6 | Better search for one compressing circuit (5-frame clip) | Proved ≤ 4 gates impossible in that family. | Nothing found. UNKNOWN, not impossible. | [CHECKPOINT_06](../CHECKPOINT_06.md) |
| C7 | Four method pilots (Life-like, second-order trees, CGP, DiffLogic long) | CGP circuits of 3–10 gates follow coarse shape on 60 frames. | They do not follow motion. | [CHECKPOINT_07](../CHECKPOINT_07.md) |
| C8 | Motion-aware CGP objective + loophole baselines | 0/32 runs followed the motion. Late "motion" was edge flicker (a metric loophole). | Small circuits fail on real motion. | [CHECKPOINT_08](../CHECKPOINT_08.md) |
| C9 | A/B/C diagnosis: teacher-forced vs autonomous; CGP vs Neural CA (NCA) | A 10,320-param continuous NCA replays a short real action from the first frame (3/3 seeds). CGP passes no stage. | Not a small rule, not compression; the model is trajectory-specific. | [CHECKPOINT_09](../CHECKPOINT_09.md) |
| C10 | Full-span diagnosis; same-size fine-tuning (contour hinge, change matching) | The "noise" was 1–2 px contour error and under-motion. Error 0.0040 (56 frames), 0.0134 (60 frames). | Gains came from more training and the contour loss; the change loss added nothing. | [CHECKPOINT_10](../CHECKPOINT_10.md) |
| C11 | Smaller models, longer playback, binary export, video→CA pipeline, local UI | 120 frames pass at 10,320 params (≈ 0.031). | 1,304 / 3,884 params fail. 240 frames (wipe + cut) fail. Files are larger than recordings. | [CHECKPOINT_11](../CHECKPOINT_11.md) |
| C12 | Phased NCAs: an external schedule with state carried across phases; automatic segmentation; explicit decision-list rules; UI inspector | 4–6 phases fix the 240-frame transition interval (0.019–0.028; a same-size single rule 0.045). Training-only state noise made long replays reproducible: 0–1467 passes (48.9 s). | Automatic segmentation was never better than uniform. The whole song without noise diverged after ~1,080 frames. Explicit rules reach ~0.10 at best. | [CHECKPOINT_12](../CHECKPOINT_12.md) |
| C13 | Whole song with the noise recipe and numpy-carried chunk state; transition-weighted loss; product UI | See [CHECKPOINT_13](../CHECKPOINT_13.md). | See CHECKPOINT_13. | [CAMPAIGN_13](../CAMPAIGN_13.md) |
