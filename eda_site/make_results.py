"""Собирает site_data/results.json из журналов экспериментов и финального аудита."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; E = ROOT / "experiments"
fs = json.load(open(E / "final_scores.json"))
ST = json.load(open(E / "stability_nofilter.json"))
NAMES = {"cat_d4": "CatBoost (depth 4)", "hybrid": "Hybrid: LightGBM boosted from logistic-regression init",
         "cat_d6": "CatBoost (depth 6)", "lgbm": "LightGBM (Optuna-tuned)", "linear": "Logistic regression on instrument levels"}
models = sorted([{"model": NAMES[k], "auc_mean": float(np.mean(v)), "auc_std": float(np.std(v))} for k, v in fs["honest"].items()],
                key=lambda r: -r["auc_mean"])
models.append({"model": "Final ensemble (equal-weight rank average, 3 seeds)", "auc_mean": fs["ensemble_oof_auc"], "auc_std": None})
hyp = pd.read_csv(E / "hypotheses.csv")
HN = {"P1_pairs_all": ("Incoming → outgoing chains (5m/1h/6h/24h, amount similarity)", "instrument order is random given composition"),
      "P1_pairs_io_only": ("Incoming → outgoing only", "same"),
      "P3_sequence_all": ("Instrument sequence: stickiness, runs, transition lift", "stickiness ≈ 1.0 (random order)"),
      "P2_structuring_all": ("Structuring: near-equal amounts (±tol)", "amounts continuous, no repeats; proxy for density"),
      "P5_detrended_levels": ("Instrument levels after removing the common trend (added)", "trend identical in both classes"),
      "P5_detrended_linear_in_ensemble": ("Detrended linear model inside the ensemble", "+0.0015 for the linear model alone, absorbed by the ensemble")}
R = {
 "models": models,
 "metric_path": [
  {"step": "Baseline aggregates (59 features)", "auc": 0.6074, "note": "LightGBM, repeated CV"},
  {"step": "+ distribution shape (49)", "auc": 0.6093, "note": "LightGBM, repeated CV"},
  {"step": "+ model zoo & ensemble", "auc": 0.6148, "note": "4-model rank average"},
  {"step": "+ instrument × direction statistics (130)", "auc": 0.6488, "note": "the key EDA insight"},
  {"step": "honest early stopping + tuning + contrast + linear/hybrid", "auc": round(fs["ensemble_oof_auc"], 4), "note": "final honest ensemble, after audit"}],
 "feature_groups": [
  {"group": "Base aggregates", "n": 59, "what": "amount stats (all / incoming / outgoing), type shares & counts, balance, activity, alert date"},
  {"group": "Distribution shape", "n": 49, "what": "kurtosis, Bowley skew, inter-quantile ranges, tail ratios"},
  {"group": "Instrument × direction statistics", "n": 130, "what": "count/mean/std/quantiles of amounts for each of 8 instrument×direction combinations"},
  {"group": "Explicit contrast", "n": 13, "what": "cash-in + card-out − bank-in − bank-out at mean / median / q25 / q75"},
  {"group": "Linear-model inputs", "n": 76, "what": "instrument levels + customer-scale / skew decomposition + log counts"}],
 "failed": [
  {"idea": "Time windows 7/30/90d, bursts, night activity, inter-arrival gaps", "delta": -0.0006, "why": "timestamps are uniform; no temporal signal"},
  {"idea": "Time features per instrument", "delta": -0.0017, "why": "same"},
  {"idea": "Distribution shape per instrument", "delta": -0.0015, "why": "signal is in location, not shape"},
  {"idea": "Amount histogram (36 bins)", "delta": -0.0012, "why": "moments already describe it"},
  {"idea": "148 pairwise ratios of instrument means", "delta": -0.0002, "why": "trees derive ratios themselves"},
  {"idea": "Transaction-level (multiple-instance) model", "delta": None, "why": "alone 0.556; hurts in blend — a single transaction cannot see the cross-instrument ratio"},
  {"idea": "Empirical-Bayes shrinkage of noisy levels", "delta": 0.0016, "why": "below threshold; shrinking toward other instruments erases the contrast"},
 ] + [{"idea": HN[r.experiment][0], "delta": float(r.delta_vs_control), "why": HN[r.experiment][1]} for r in hyp.itertuples()]
   + [{"idea": "Replace all amounts by detrended amounts", "delta": -0.0113,
       "why": "all of the loss comes from min / lower-quantile features: raw minima sit in late & burst transactions (legitimate pre-alert info)"}],
 "audit": {
  "post_date_note": "840 train / 51 test transactions are dated later than 00:00 of signal_sanasi (21 / 2 alerts). signal_sanasi has no time; for these alerts the whole 180-day window is shifted by one day and the pre-alert burst ends just before 24:00 of the same date — pre-alert history, kept. signal_sanasi is stored as a calendar date without time, so the alert moment is inferred from the data structure",
  "permutation_auc_range": [0.479, 0.509], "adversarial_auc": ST["adversarial_auc"],
  "contrast_half_A": "Bank in −0.48, Cash in +0.44, Bank out −0.40, Card out +0.25",
  "contrast_half_B": "Bank out −0.40, Cash in +0.38, Bank in −0.36, Card out +0.32",
  "splits": [{"split": s, "final_gap": fg, "control_gap": cg} for s, fg, cg in
             [(31337, -0.0122, -0.0018), (1, 0.0203, 0.0176), (2, 0.0251, 0.0183), (3, 0.0031, 0.0260), (4, 0.0046, 0.0141), (5, 0.0192, 0.0196), (6, 0.0150, 0.0160), (7, -0.0108, -0.0177)]],
  "selection_bias": 0.0035, "selection_bias_se": 0.0036,
  "residual_features_above_noise": 1, "residual_features_tested": 371, "residual_expected_by_chance": 17,
  "stability_single_seed_mean": ST["single_mean"], "stability_single_seed_std": ST["single_std"], "stability_rank_spearman": ST["spearman"],
  "stability_top1000_overlap": ST["top1000"], "leave_one_out_max_drop": ST["loo_max_drop"],
  "date_auc_max": 0.508},
 "ceiling": {"fraction": [0.125, 0.25, 0.5, 0.75, 1.0], "auc": [0.5895, 0.6069, 0.6252, 0.6310, 0.6398], "limit_low": 0.652, "limit_high": 0.657},
 "junk_feature_ranks": [42, 44, 71], "junk_total": 111,
}
json.dump(R, open(Path(__file__).resolve().parent / "site_data" / "results.json", "w"), indent=1, ensure_ascii=False)
print("results.json ok, ensemble", round(fs["ensemble_oof_auc"], 5))
