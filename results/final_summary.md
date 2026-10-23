# Final summary

**Designated final submission:** `submissions/submission_rerank_v5.csv`.
**Kaggle public/private:** **0.44501 / 0.43419** (final feedback supplied by the user).

## Validation results

Fit window A: `[2020-03-09, 2020-05-18)`; validation window B: `[2020-05-18, 2020-07-27)`; 6,376 fixed candidates and 1,260 template users. TK evaluates 1,238 users; T1 evaluates 1,246.
Scores and runtimes below come from `validation_methods_v4.csv` (15 methods), the selected row of `validation_rerank.csv` (rerank_v5), and the frozen standalone score50 evaluation in `component_check.txt`. No configuration search was run for these reports.
Runtime is the recorded per-configuration time: fixed-model rows include their measured fit/score/evaluation; hybrid rows cover fusion/ranking/evaluation with shared fits separate; rerank_v5 includes the internal-A classifier fit and B prediction/evaluation, with feature preparation separate. The score50 runtime includes score generation, ranking and TK/T1 evaluation after shared counts; its preparation is timed separately in `component_check.txt`. All seeds are 42; the classifier uses `early_stopping=False`.

| Method | TK MAP@5 | T1 MAP@5 | Saved runtime (s) | Frozen hyperparameters |
|---|---:|---:|---:|---|
| popularity | 0.261874663 | 0.737594971 | 0.244 | `{}` |
| popularity_freq | 0.295437309 | 0.610337079 | 0.156 | `{"score":"all_fit_users_completed_80_airings"}` |
| rewatch | 0.257432014 | 0.676803103 | 0.047 | `{}` |
| rewatch_days | 0.321115374 | 0.768651685 | 0.158 | `{"tie_break":"seconds_then_id"}` |
| MF | 0.217987345 | 0.674007491 | 0.897 | `{"alpha":1,"factors":4,"iterations":10,"reg":10,"seed":42}` |
| PPR | 0.293149345 | 0.774063670 | 25.493 | `{"alpha":0.8,"batch_size":128,"max_iterations":500,"tolerance":1e-06}` |
| CBF | 0.018633549 | 0.187332798 | 0.311 | `{"idf":"smooth_A_only","metadata_scope":"A_only","profile_weight":"log1p(seconds/60)","tf":"binary","year":"first_year_decade"}` |
| association rules | 0.281963292 | 0.737372927 | 1.270 | `{"lift_strictly_greater_than":1,"minimum_support":20,"score":"max_confidence"}` |
| slot | 0.276980120 | 0.750524345 | 0.159 | `{"slot":"channel_weekend_hour","viewing":"all_fit_views"}` |
| exposure | 0.327187220 | 0.778017657 | 0.140 | `{"smoothing_days":3}` |
| score80 | 0.395503276 | 0.716698769 | 0.051 | `{"k":80}` |
| score50 | 0.404794247 | 0.743905832 | 0.067 | `{"k":80}` |
| hybrid_old | 0.306975184 | 0.781233280 | 0.183 | `{"MF":2.0,"PPR":2.0,"popularity":0.0,"rewatch":0.5}` |
| hybrid_v2 | 0.333266694 | 0.790387908 | 0.144 | `{"rewatch_days":1.0,"w_mf":0.2,"w_pop":0.0,"w_ppr":0.5}` |
| hybrid_v3 | 0.337903653 | 0.808533975 | 0.236 | `{"w_days":1.0,"w_exp":2.0,"w_mf":0.2,"w_pop":0.3,"w_ppr":2.0,"w_slot":2.0}` |
| hybrid_v4 | 0.404992371 | 0.743258427 | 0.083 | `{"k":80,"w50":0.5,"w_pop":0.1,"w_ppr":0.05,"w_slot":0.05}` |
| rerank_v5 | 0.414640325 | 0.776126271 | 12.136 | `{"learning_rate":0.05,"max_iter":300,"max_leaf_nodes":7}` |

CBF is the final availability-only comparator: fitting-window metadata supplies both history and candidate vectors; unknown content is zero. `score80` uses its TK-selected k=80. The v4 score80 coefficient is fixed at 1.

## Kaggle submissions and lift over popularity

Kaggle scores are user-provided feedback. Lifts are computed independently against popularity in the same column; they are not cross-metric score ratios.

| Submission | Validation TK | TK lift | Kaggle public | Public lift | Kaggle private | Private lift |
|---|---:|---:|---:|---:|---:|---:|
| popularity | 0.261874663 | 1.000000x | 0.28078 | 1.000000x | 0.27750 | 1.000000x |
| hybrid_old | 0.306975184 | 1.172222x | 0.32064 | 1.141962x | 0.30256 | 1.090306x |
| hybrid_v2 | 0.333266694 | 1.272619x | 0.35814 | 1.275518x | 0.33353 | 1.201910x |
| hybrid_v3 | 0.337903653 | 1.290326x | 0.36420 | 1.297101x | 0.33702 | 1.214486x |
| hybrid_v4 | 0.404992371 | 1.546512x | 0.43024 | 1.532303x | 0.41795 | 1.506126x |
| rerank_v5 | 0.414640325 | 1.583354x | 0.44501 | 1.584906x | 0.43419 | 1.564649x |

Final-file SHA256: `c598594332dae6f1b6de4ec0995ba783c24aeb6a444a0658fa3d9cbbb3b0610d`.
