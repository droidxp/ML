# Second Study — DroidXPflow on the blind spots of the MAS approach

Replication package for the second study of the extended version of our ECOOP
2025 paper. It reproduces **every number** of that section (Section 5): Tables 8–13,
Figure 7, and Findings 5–7.

The study asks one question: when the MAS approach fails to flag a repackaged
app as malware, is that a limitation of *dynamic analysis*, or only of the
*observable* it mines? To answer it, we apply DroidXPflow — which mines the
sandbox from network flows instead of calls to sensitive APIs — to `FocusDS`,
the subset of `LargeDS` built from precisely the samples the MAS approach
misses most.

## Quickstart

```bash
./run_all.sh
```

That creates a virtualenv with the pinned dependencies, runs the pipeline, and
checks the result against the numbers printed in the paper. Expect about a
minute on 8 cores. A successful run ends with:

```
  [OK  ] Sec. 5.1  FocusDS       14/14 values match
  [OK  ] Sec. 5.3  Protocol      7/7 values match
  [OK  ] Table 8   Hyper-params  7/7 values match
  [OK  ] Table 9   Algorithms    50/50 values match
  [OK  ] Table 10  MAS vs flow   11/11 values match
  [OK  ] Table 11  Per family    8/8 values match
  [OK  ] Table 12  Importances   4/4 values match
  [OK  ] Figure 7  Venn          4/4 values match
  [OK  ] Table 13  Similarity    10/10 values match

All 115 values reproduce the paper.
```

Add `--figures` to also regenerate `vennFocusTP.pdf` and `similarityBands.pdf`
(set `FOCUS_FIGS` to write them straight into the paper directory).

## Inputs

Both ship with this repository; nothing needs to be downloaded.

| File | Where | What it is |
|---|---|---|
| `Final_file.zip` → `Final_file.csv` | repository root | The 1,596 network-flow features per repackaged app. Extracted automatically on first run. |
| `large_ds.csv` | repository root | The `LargeDS` labels of the first study: malware family, `apidetected` (the MAS verdict), and the Similarity Score. |

Override the locations with `FOCUS_DATA` and `FOCUS_ML` if you keep them
elsewhere.

### How to read a feature name

Feature columns are named `<cicflowmeter_feature>_<statistic>_<port>`:

- **76** base features produced by CICFlowMeter, minus the eight that identify
  the execution rather than characterize it;
- **7** statistics aggregating all flows of one app —
  `count`, `min`, `max`, `mean`, `median`, `var`, `skew`;
- **3** destination ports — `53` (DNS), `80` (HTTP), `443` (HTTPS).

76 × 7 × 3 = **1,596**. So `bwd_byts_b_avg_median_443` is the median, over an
app's HTTPS flows, of the average bulk byte rate in the backward direction —
the feature that dominates Table 12.

## Outputs

| File | Contents |
|---|---|
| `focus_results.json` | Every metric of the study, plus the environment it ran in |
| `focus_similarity.json` | Similarity Score bands and Spearman correlations (RQ5) |
| `focus_predictions.csv` | Per-app test-split predictions of the selected model |

## Where each number in the paper comes from

`verify.py` encodes this mapping, so it doubles as an index.

| Paper | `focus_results.json` key |
|---|---|
| §5.1, FocusDS composition | `dataset` |
| §5.1, MAS baseline on FocusDS | `mas_on_focus_full`, `mas_per_family_full` |
| §5.3, train/test split | `split` |
| Table 8, hyper-parameters | `cv.best_params` |
| Table 9, algorithm comparison | `algorithms` |
| Table 10, MAS vs DroidXPflow vs Combined | `test_comparison` |
| Table 11, detection per family | `per_family_test` |
| Table 12, feature importances | `top_features` (and `selected_features` for the selection step) |
| Figure 7 and Finding 6, Venn | `contribution` |
| Table 13 and Finding 7 | `similarity_bands`, `spearman_*` in `focus_similarity.json` |

## What the pipeline does

`focus_study.py`

1. Deduplicates `LargeDS` by APK hash. Nine of its 4,076 records duplicate an
   APK already in the dataset (three `gappusin`, two `androrat`, one each
   `smsreg`, `droidkungfu`, `kuguo`; all nine are malware the MAS approach
   classifies correctly), leaving **4,067** distinct apps. This keeps an app
   from landing in both the training and the test split.
2. Builds **FocusDS**: every `gappusin`/`revmob` malware plus every non-malware
   repackaged app — 2,722 apps, 1,541 malware and 1,181 non-malware.
3. Splits 70/30, stratified, `random_state=0`, **before anything is fitted**.
4. Selects the 20 most important features by Gini importance of a default
   Random Forest, fitted **on the training split alone**.
5. Tunes all seven algorithms by grid search with 5-fold stratified
   cross-validation, again **on the training split alone**, optimizing F1.
6. Selects the model to report by its **cross-validated F1** (Table 8), so the
   choice never depends on the test set.
7. Evaluates each on the held-out test split at the **standard decision rule**
   (probability ≥ 0.5, or the natural SVM boundary), so the seven are compared
   at a common operating point.
8. Writes the metrics and the predictions of the selected model.

`focus_similarity.py` reads those predictions — it does **not** refit — so the
RQ5 analysis necessarily describes the same model as the rest of the section.
`make_figs.py` likewise.

### Two modelling choices worth knowing

- **Missing values.** CICFlowMeter leaves a statistic undefined when an app
  produced no flow on a port (the skewness of a single flow, say). 3.8% of
  cells are affected and 3,220 of the 4,067 apps have at least one. They are
  mapped to a large negative sentinel rather than to zero, so that "no traffic
  observed" stays distinguishable from "a measured value of zero".
- **Labels.** `malicious` in `Final_file.csv` and `malware` in `large_ds.csv`
  agree on all 4,067 apps, so the ground truth is unambiguous; the code uses
  the former.

## How this differs from the scripts at the repository root

Those scripts belong to the earlier DroidXPflow study, which evaluated on the
whole dataset. Three differences matter, and each one *lowers* the reported
performance by removing an optimistic bias:

| | Root scripts | Here |
|---|---|---|
| Feature selection | over the complete dataset (`clearFile.py`) | training split only |
| Hyper-parameters | fixed values tuned on the complete dataset | re-tuned by CV inside the training split |
| Decision threshold | 0.4 for XGBoost, 0.5 for the rest | 0.5 for all seven |
| Duplicate APKs | kept | removed before splitting |

**The numbers here are therefore not comparable to those of the root scripts**,
and are not meant to be: this study evaluates on an adversarial subset chosen
to be hard, not on a dataset chosen to be favourable.

## Environment

The results were produced with Python 3.14.3 and the versions pinned in
`requirements.txt` (scikit-learn 1.9.1, xgboost 3.4.1, pandas 3.0.6, numpy
2.5.3, scipy 1.18.1). Every run records its own environment under the
`environment` key of `focus_results.json`.

The pipeline is deterministic: repeated runs on the same versions produce
byte-identical output. Across library versions it may not be, because estimator
defaults and solvers change between releases — `verify.py` reports exactly
which values moved, so a near-miss is easy to tell apart from a real
divergence.
