# Restore of Campaign 04 files (2026-11-09)
- Deleted from the work tree at 2026-11-09 10:25:56 KST (by the user's rollback of a different session, per user).
- State before restore: worktree_files_before_restore.txt, worktree_sha256_before_restore.txt, diff_snapshot64277b_vs_worktree_before_restore.patch.
- Code restored ONLY for the 6 missing files, copied byte-for-byte from runs/frozen/64277b4ec640 (frozen 2026-11-09 ~00:33 KST).
  Their sources are identical to the snapshot the searches ran from (9b8f0db4a7ba), except exp11_reduce.py, which adds the
  documented `target` fallback for exp07/exp08 results (KeyError fix, runs/exp11_crash_D1_no_target.log).
- docs/CAMPAIGN_04.md: ORIGINAL text, not a rewrite. It was recovered from this session's transcript, out of the output of a
  `cat docs/CAMPAIGN_04.md` run at 2026-11-09 00:31 KST. The file's mtime was then 00:07:18 KST, before the main runs
  started (00:07:35). The recovered file is 7842 bytes, the same size as the original's directory listing. Byte identity
  beyond size cannot be proven (no other copy exists).
- runs/exp11b (created 02:10 KST): origin not established (no code/config/log in this session produced it). Kept, EXCLUDED
  from official results.
