# Second study (DroidXPflow on the MAS blind spots)

Reproduces every number reported in Section 6 of the paper.

## Inputs

- `Final_file.csv` — the 1,596 network-flow features per repackaged app
  (unzip `ML/Final_file.zip` from the DroidXPflow replication package,
  <https://github.com/droidxp/ML>).
- `large_ds.csv` — the LargeDS labels of the first study (family, `apidetected`
  i.e. the MAS verdict, similarity), same repository.

`Final_file.csv` is read from this directory by default; point `FOCUS_DATA` at
another one if you keep it elsewhere. `large_ds.csv` is read from the path in
`FOCUS_ML`.

## Running

    python3 -m venv venv && ./venv/bin/pip install scikit-learn xgboost pandas numpy scipy matplotlib matplotlib-venn
    FOCUS_DATA=/path/to/csvs ./venv/bin/python focus_study.py  # -> focus_results.json, focus_predictions.csv
    ./venv/bin/python focus_similarity.py                      # -> focus_similarity.json
    ./venv/bin/python make_figs.py                             # -> vennFocusTP.pdf, similarityBands.pdf

Run them in this order: the last two consume what `focus_study.py` writes.
The pipeline is deterministic — repeated runs produce byte-identical JSON.

## What the scripts do

`focus_study.py` deduplicates LargeDS by APK hash (nine records of the first
study duplicate an APK already in the dataset), builds FocusDS (every
gappusin/revmob malware plus every non-malware repackaged app), splits it 70/30
stratified with `random_state=0`, and then, **using the training split only**:

1. selects the 20 most important features by Gini importance of a default
   Random Forest;
2. tunes every algorithm by grid search with 5-fold stratified
   cross-validation, optimising F1.

It then evaluates the seven tuned algorithms on the held-out test split, each
with the standard decision rule (probability >= 0.5, or the natural SVM
boundary), so that the comparison is at a common operating point. It reports
every metric, the selected hyper-parameters, the per-family breakdown, and the
MAS/DroidXPflow contribution analysis, and writes the test-split predictions of
the selected model to `focus_predictions.csv`.

Two deviations from the published replication package of DroidXPflow, both of
which lower the reported numbers but remove an optimistic bias:

- the package selects the 20 features over the *complete* dataset, which leaks
  test information into the selection; here the selector is fitted on the
  training split alone;
- the package reuses hyper-parameters and a 0.4 decision threshold tuned on the
  complete dataset; here every hyper-parameter is re-tuned inside the training
  split, and all algorithms are compared at the same threshold.

`focus_similarity.py` reads `focus_predictions.csv` — so it reports on exactly
the model selected by `focus_study.py`, rather than refitting its own — and
reports the detection rate per Similarity Score band plus the Spearman
correlation between the Similarity Score and a correct classification, for both
approaches.
