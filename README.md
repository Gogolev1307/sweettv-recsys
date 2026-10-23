# sweet.tv TV-programme recommender

A reproducible recommendation project for Kaggle's **sweet.tv - TV program
recommender**, evaluated by **MAP@5**. It progresses from popularity and
collaborative/content methods to personal habits, channel-aware completion, and
candidate generation followed by learned re-ranking.

## Designated final submission

**[`submissions/submission_rerank_v5.csv`](submissions/submission_rerank_v5.csv)**
is the designated final submission. Final user-provided Kaggle feedback is
**public 0.44501 / private 0.43419**. Its A-to-B validation scores are
**TK MAP@5 0.414640325 / T1 MAP@5 0.776126271**.

- [Final comparison and six-submission Kaggle calibration](results/final_summary.md)
- [Final validation examples, including popularity-fill provenance](results/examples_final.md)
- [Step-by-step AP@5 hand check](results/hand_check.txt)
- [Re-ranker validation results](results/validation_rerank.csv)
- [Frozen selected configuration](results/rerank_v5_selected.json) and
  [final-refit provenance](results/rerank_v5_final.json)

## Data and environment

Raw data is **not included**. Download the competition files from Kaggle into
`data/`, keeping these names:

```text
data/dataset11-30.csv
data/export_arh_11-20-final.csv
data/export_arh_21-30-final.csv
data/export_arh_31-42-final.csv
data/submission.csv
```

`data/` is read-only to the project and git-ignored. Intermediate typed data,
joins, profiles and training feature rows are fingerprinted in git-ignored
`cache/`. Reports are in `results/`; Kaggle CSVs are in `submissions/`.

Environment used: **Python 3.12.4**, NumPy **1.26.4**, pandas **2.2.2**, SciPy
**1.13.1**, scikit-learn **1.4.2**. For the same Python environment:

```bash
python3 -m pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-learn==1.4.2
```

All stochastic fits use seed **42**. Runners set BLAS/OpenMP to one thread
before importing NumPy; this also avoids overhead in small exact ALS solves.

## Published ground truth and TK validation

The competition's definition, quoted verbatim as supplied by the user:

> all entries with tv_show_id = 0 were removed from TV program schedule files; for each user and each TV show user screen time (time user watched the show divided by total duration of the show) was calculated. Only TV shows with high screen time (>= 0.8) were taken into account; for each user high-screen time TV shows frequency was calculated; for each user strictly 5 most frequent TV shows were selected DESC

**TK** implements the requested airing-level version of that rule in the label
window:

1. Aggregate overlap seconds per `(user_id, airing_id)` for labelled airings.
2. Screen time is `overlap_seconds / full_schedule_duration`.
3. Keep airings with screen time **>= 0.8**, and count them per user/programme.
4. Take the five programmes with the largest counts, breaking ties by ascending
   programme id. A user's relevant set has `min(5, qualifying programmes)` items.

There is **no 300-second minimum** for TK or completion counts: a short airing
can qualify at 80%. On B, **1,260 = 1,238 evaluated + 22 with empty TK**; the
evaluated median relevant-set size is 5. Discovery of the published rule replaced
the earlier proxy truths as the final selection objective. TK lifts closely
tracked the supplied Kaggle feedback; v4 and the re-ranker select by TK MAP@5.

The historical truths remain in saved reports: **T1** is any >=300-second airing;
**T2** is >=3,600 total positive labelled seconds per programme; **T3** requires
eligible >=300-second airings on at least five actual viewing dates; **T4** takes
the ten programmes with most total labelled seconds. T1 evaluates 1,246 users
and has median 110 relevant programmes among evaluated users.

AP@5 is the sum of precision at relevant recommendation ranks, divided by
`min(5, number of relevant items)`. Repeated predictions receive no second
credit and retain their list rank. MAP averages only users with nonempty truth.

## Loading, interval joins and thresholds

- `user_id` is loaded as **uint64**. View timestamps use `YYYY-MM-DD HH:MM:SS`;
  schedule timestamps use `DD.MM.YYYY HH:MM:SS`. `vsetv_id` is `channel_id`.
  View and schedule clocks align; no timezone shift is applied.
- `duraton` is misspelled in the source and describes viewing, not programme
  length. Viewing seconds come from **`stop_time - start_time`**; programme
  duration comes from the schedule's `duration`.
- Concatenate training schedules and deduplicate **`(channel_id, start_time)`**,
  keeping the later row. The files share the May 18 calendar boundary; actual
  deduplication/input checks are printed by `run_validation.py`.
- Join a view to **every** same-channel overlapping airing, using half-open
  intervals `[start, end)`. Each airing receives
  `max(0, min(view_stop, airing_end) - max(view_start, airing_start))` seconds.
  Views spanning several airings contribute to all of them.
- Clip viewing to the fitting/label window before aggregation. Sum sessions per
  user/airing, then apply the required threshold. Sessions are additive; their
  summed fraction can exceed one, but a completed user/airing still counts once.
- The **300-second** rule defines the common seconds matrix, ordinary watched
  histories, distinct-viewer popularity, and cold start. The **80% / 50%** rules
  define completion statistics independently of that seconds minimum.
- **`tv_show_id=0` is never an item, truth id, or submission id.** Its viewing is
  reported in join coverage and retained in raw-view slot habits, which use all
  fitting viewing including short sessions and unknown/unlabelled channels.

## Fixed temporal protocol

All boundaries below are half-open:

| Role | Interval |
|---|---|
| Fit A, weeks 11-20 | `[2020-03-09, 2020-05-18)` |
| Validate/label B, weeks 21-30 | `[2020-05-18, 2020-07-27)` |
| Final fitting, weeks 11-30 | `[2020-03-09, 2020-07-27)` |
| Test target, weeks 31-42 | `[2020-07-27, 2020-10-19)` |

- Ordinary validation methods fit A only. B viewing supplies evaluation labels
  and scores. The validation pool is the **6,376 labelled programme ids with
  airing starts inside B** in the combined, deduplicated training schedule.
- The final pool is all **6,636 labelled ids in the test schedule**. Target
  schedules supply only programme/channel/time availability: `tv_show_id`,
  `channel_id`, `start_time`, `duration`. No target content metadata enters final
  comparison features. Numeric availability includes known export-boundary
  carry-in and is clipped to the target interval.
- Truth is not filtered to candidates, including boundary carry-in cases.
  **Re-watching is allowed**: previously watched programmes remain eligible.
- Every user gets **five distinct candidate ids**. Score ties use ascending
  programme id; the standalone day reference additionally breaks day-count ties
  by watched seconds before id.
- Short method proposals are filled from fitting-window distinct-viewer
  popularity. Users with **no >=300-second labelled fitting airing** receive
  that same popularity list for every method, even if another feature has raw
  viewing/completion evidence. There are 10 A-cold test users and 2 full-training
  cold users.
- Final fitting combines overlaps across the A/B boundary **before** applying
  its thresholds. CSV writers preserve the template's 1,260-user order, write
  plain integer user ids, and validate five space-separated distinct positive
  test-candidate ids. Existing submission CSVs are never overwritten.

## Methods

| Method | Definition / frozen setting |
|---|---|
| popularity | Number of distinct fitting users with a >=300-second airing of the programme; also the shared fill/cold-start list. |
| popularity_freq | Total fitting user/airings completed >=80%, summed across users. |
| rewatch | The user's total seconds from >=300-second fitting airings. |
| rewatch_days | Distinct actual viewing dates on qualifying >=300-second airings; ties by seconds, then id. |
| MF | NumPy implicit ALS: preference 1 for observations, confidence `1 + alpha*log1p(seconds/60)`; 10 exact user/item solve iterations; factors=4, reg=10, alpha=1. |
| PPR | Bipartite log-seconds graph, row-normalized in both directions; follow probability 0.8, one-hot restart, dangling mass re-injected through restart; L1 tolerance 1e-6, max 500 iterations, batches of 128. |
| CBF | Binary-TF/smoothed-IDF category, genres, comma-separated director/actors, production decade; L2 item vectors and log-seconds weighted cosine profile. Final comparisons use A-only metadata; unknown content is zero. |
| association rules | Programme-set baskets; pair support >=20 users, lift >1; candidate score is maximum applicable rule confidence. |
| slot | All raw fitting seconds per `(channel, weekend flag, hour)`, split at hour boundaries and row-normalized; dot product with target-airing seconds per slot. |
| exposure | Watched days divided by `(fitting airing days + 3)`, multiplied by target airing days; dates count once across channels/repeats. |
| score80 | Channel-aware completion habit `sum_c n80(u,p,c)/(nA(p,c)+80)*nT(p,c)`; standalone best k=80. |
| hybrid_old | `.5*M(rewatch) + 2*M(PPR) + 2*M(MF)`; popularity weight 0. |
| hybrid_v2 | `M(rewatch_days) + .5*M(PPR) + .2*M(MF)`; popularity weight 0. |
| hybrid_v3 | `M(days) + 2*M(exposure) + 2*M(slot) + 2*M(PPR) + .2*M(MF) + .3*popularity/max`. |
| hybrid_v4 | Frozen completion fusion below, selected by TK. |
| rerank_v5 | Top-50 completion candidates followed by the frozen 18-feature classifier below; designated final submission. |

`M` is per-user min-max normalization over the entire candidate pool; a constant
vector becomes zero. Popularity normalization divides by the candidate-pool
maximum. Calendar features split actual viewing/airing intervals at midnight;
half-open stops do not add an extra date.

### Frozen v4 formula

For each fitting user/programme/channel, `n80` and `n50` count airings reaching
80% and 50% screen time; `nA` and `nT` count unique labelled channel/start
airings intersecting the fitting and target windows. Full airing duration is
the screen-time denominator. With **k=80**:

```text
score80 = sum_channels n80 / (nA + 80) * nT
score50 = sum_channels n50 / (nA + 80) * nT
v4 = M(score80) + 0.5*M(score50) + 0.05*M(slot)
     + 0.05*M(PPR) + 0.1*popularity_freq/max(popularity_freq)
```

A new target-only programme/channel pair has no learned personal completion
rate. The exact published quote, grid results and selected coefficients are in
`results/hybrid_v4_selected.json` and `results/validation_methods_v4.csv`.

### Final rerank_v5

Candidate generation uses each user's top 50 within the target pool, ordered by

```text
M(score80) + 0.5*M(score50) + 0.1*popularity_freq/max(popularity_freq)
```

The **18 features** are:

1. For each of >=80% screen time, >=50% screen time, and >=300-second overlap:
   airing count `n`; channel-aware `s=sum n/(nA+80)*nT`; `nt=sum nT` on that
   programme's channels where the user met the threshold; elapsed days from the
   fit cutoff to the last such airing start, or 999 if none (12 features).
2. Programme popularity_freq; all-channel target airing count; the user's total
   >=80% fitting airings across all programmes; that programme's share of the
   total; candidate base score; zero-based base rank 0-49 (6 features).

User/programme ids are indexing keys. Labels indicate membership in the label
window's TK. **HistGradientBoostingClassifier** uses **max_iter=300,
learning_rate=0.05, max_leaf_nodes=7, random_state=42, early_stopping=False**.
Predicted-probability ties use ascending programme id.

Validation classifier training uses only two splits inside A:

| Fit interval | Label interval |
|---|---|
| `[2020-03-09, 2020-04-06)` | `[2020-04-06, 2020-05-18)` |
| `[2020-03-09, 2020-04-20)` | `[2020-04-20, 2020-05-18)` |

All users with >=300-second fitting history contribute 50 candidate rows,
including negative rows for users with empty label truth: 477,000 rows total.
The three prescribed classifier configurations were compared on B TK. The
selected model reached TK 0.414640325 versus v4's 0.404992371 and won all 200
seeded half-user comparisons and their complementary halves.

The final classifier refits the frozen configuration on exactly these two
weeks-11-30 splits (484,200 rows):

| Fit interval | Label interval |
|---|---|
| `[2020-03-09, 2020-05-18)` | `[2020-05-18, 2020-07-27)` |
| `[2020-03-09, 2020-06-15)` | `[2020-06-15, 2020-07-27)` |

It then generates features from all weeks-11-30 fitting interactions, scores
the test candidates, and writes `submission_rerank_v5.csv`. No test viewing or
test labels are used. The designated final file is preserved as generated.

## Reproduction commands and measured runtimes

Run from the repository root after installing the environment and downloading
the data. This order reproduces the saved model selections and submission chain:

| Order | Command | Historical measured runtime |
|---:|---|---:|
| 1 | `python3 run_validation.py` | about 5 s cached; 39 s uncached |
| 2 | `python3 run_methods.py` | 148 s |
| 3 | `python3 make_submission.py` | 58 s |
| 4 | `python3 run_methods_v2.py` | 37 s |
| 5 | `python3 make_submission.py --v2` | 47 s |
| 6 | `python3 run_methods_v3.py` | 162 s |
| 7 | `python3 make_submission.py --v3` | 50 s |
| 8 | `python3 run_methods_v4.py` | 105 s |
| 9 | `python3 make_submission.py --v4` | 43 s |
| 10 | `python3 run_rerank.py` | 109 s, including validation and final refit |

These are measured historical pipeline times on the environment above; cache
state and hardware affect elapsed time. Per-method saved runtimes in the final
summary have their own scopes: hybrid fusion excludes shared component fits,
and the re-ranker configuration time excludes shared feature preparation.

`src/data.py`, `src/join.py` and `src/evaluate.py` define the shared loading,
interaction and metric contracts. Experiment reports and frozen JSON manifests
record configurations, seeds, time windows, package versions and source hashes.
The three final presentation reports use saved scores and fixed validation-fit
replays for example predictions, with explicit popularity-fill provenance. They
record the final Kaggle feedback supplied by the user.
