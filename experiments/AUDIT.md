# Final audit — model frozen

Final model = version **without any date filter**: OOF ROC-AUC **0.65334**,
`submissions/team_3B832E89.csv` md5 `a0d1cf0afc9be6c039ad19590cfd26c9` (reproduced byte-identically by
`train_final.py` and `notebooks/solution.ipynb`).

## Pipeline checks
| check | result |
|---|---|
| Target in features | no — `assert TARGET not in columns`; permutation test: shuffled labels → 0.48–0.51 |
| Grouping by `signal_id` | all builders `groupby(signal_id)` + `reindex(signals order)`; index order/uniqueness asserted |
| CV integrity | asserted per fold: train / early-stop / validation disjoint; each row validated exactly once per seed |
| Train/test preprocessing | same functions; instrument norms & linear-model medians from train only; identical column order asserted; adversarial validation AUC **0.493** |
| Caches | final pipeline builds everything from raw files; reads only `experiments/best_lgb.json` (frozen params) |
| Train/test ID overlap, shared transactions | none |

## Transactions later than `signal_sanasi` — NOT leakage, used
840 train transactions (21 alerts) and 51 test transactions (2 alerts) have `tranzaksiya_vaqti > signal_sanasi`.
- `signal_sanasi` is a date without time (00:00:00 for all 20 000 alerts); the alert moment within that day is not given.
- 99 % of alerts: 180-day history ending with a ~3-minute burst just before 00:00 of the signal date.
- The 21 + 2 alerts: **all** their post-00:00 transactions form one 3-minute burst ending just before 24:00 of the same
  date (e.g. 139 of 140 in one 3-minute window); their whole window is shifted by a day (starts ≤ 178.95 days before the
  date, span ≤ 180 days); only 1 of 21 has a burst before 00:00. ⇒ the alert moment is the end of that day and these are
  the pre-alert trigger transactions.
- Removing them would delete these alerts' burst and cut their window to ~179 days. For reference 6/21 escalated —
  with n = 21 this proves neither presence nor absence of leakage; the decision rests on the data structure only.
  `signal_sanasi` is stored as a calendar date (`YYYY-MM-DD` in the raw CSV); the exact alert time is not provided in
  the data or the task description, so the alert moment is inferred. Same pattern in test; identical treatment.
- An audit step had briefly filtered them (CV 0.65347, prediction Spearman 0.99987 vs final) — this was reverted.

## Trend (detrending amounts costs −0.011)
- Calendar effect on amounts ≈ 0 (−0.02/year); date features alone AUC ≤ 0.508; removing them −0.001 (unstable)
  → no date–target dependence.
- The whole −0.011 comes from min / lower-quantile features (26 features: −0.0111; other families ≈ 0 or positive):
  amounts decline toward the alert, so raw minima come from late / burst transactions. Timing-only features do not
  recover the loss. Pre-alert information, consistent in test → legitimate, kept.

## Stability (fixed 5-model ensemble, no filter)
- Single-seed ensemble over seeds [42, 202, 777, 1001, 2024]: 0.6512, 0.6513, 0.6544, 0.6503, 0.6528 →
  **0.6520 ± 0.0014**; 3-seed bag (final) 0.65334.
- Test ranking between seeds: Spearman 0.9956 (min 0.9946); top-1000 overlap 94.2%.
- Leave-one-model-out: max drop 0.0011 (LightGBM); no single component carries the result.
- The linear model is near the 0.640 selection threshold (0.6413 on final seeds, 0.6397 on extra seeds);
  the composition is frozen at 5 models (removing it: -0.0002).

## Reproducibility
- `train_final.py` (with fold/target asserts) reproduces per-model 0.64468 / 0.64769 / 0.64918 / 0.64744 / 0.64126,
  ensemble 0.65334, and the repository submission byte-for-byte.
- `notebooks/solution.ipynb` contains the same code copied verbatim from `src/`; its submission is byte-identical.

## Submission
13/13 format checks: columns `signal_id,ehtimollik`, 6000 rows, order = test_signals, unique IDs, no NaN/inf,
values in [0, 1], no index column. File name `team_3B832E89.csv` (TEAM_ID 3B832E89).

## Controlled experiments after the audit (model unchanged)
| question | result | decision |
|---|---|---|
| Refit on full outer-train (80%) with best_iteration from tr2/es | ensemble 0.65299 vs 0.65334 (Δ −0.00035; per seed +0.0013 / −0.0003 / −0.0007; paired bootstrap 95% CI [−0.0013, +0.0005]) | keep current scheme |
| P5 detrended linear model in the ensemble | 0.65371 vs 0.65334 (Δ +0.00037; paired bootstrap 95% CI [−0.0001, +0.0008]) | keep current model |

**Model finally frozen: OOF ROC-AUC 0.65334, submission md5 a0d1cf0afc9be6c039ad19590cfd26c9.**
