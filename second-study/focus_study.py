"""
Second study of the ECOOP extension: DroidXPflow restricted to the malware
families for which the MAS approach failed most significantly in the ECOOP
paper (gappusin, revmob) plus every non-malware repackaged app.

Outputs a JSON blob with every number the paper needs.
"""
import json
import os
import warnings
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

HERE = dirname(abspath(__file__))


def _locate(filename, override):
    """Find an input file: $override, then next to this script, then one level
    up (which is the repository root when these scripts live in a subdirectory
    of the replication package)."""
    if override:
        return override
    for candidate in (HERE, dirname(HERE)):
        if os.path.exists(join(candidate, filename)):
            return candidate
    raise SystemExit(
        f"{filename} not found in {HERE} or {dirname(HERE)}. "
        f"Unzip Final_file.zip and/or set FOCUS_DATA / FOCUS_ML."
    )


# Final_file.csv comes from unzipping Final_file.zip; large_ds.csv ships with
# the replication package.
DATA = _locate("Final_file.csv", os.environ.get("FOCUS_DATA"))
ML = _locate("large_ds.csv", os.environ.get("FOCUS_ML"))
OUT = HERE
FOCUS_FAMILIES = ["gappusin", "revmob"]
SEED = 0
CV_FOLDS = 5

# ---------------------------------------------------------------- load & join
flow = pd.read_csv(join(DATA, "Final_file.csv"))
large = pd.read_csv(join(ML, "large_ds.csv"))

large = large.drop_duplicates(subset="sha256", keep="first")
flow = flow.drop_duplicates(subset="hash", keep="first")

df = flow.merge(
    large[["sha256", "family", "malware", "apidetected", "similarity"]],
    left_on="hash",
    right_on="sha256",
    how="inner",
)
df["family"] = df["family"].fillna("None").replace("None", "NoFamily")
df["mas"] = df["apidetected"].astype(bool)
df["y"] = df["malicious"].astype(int)

# ------------------------------------------------------------ build the focus
is_focus_malware = (df["y"] == 1) & (df["family"].isin(FOCUS_FAMILIES))
is_benign = df["y"] == 0
focus = df[is_focus_malware | is_benign].reset_index(drop=True)

meta_cols = ["hash", "sha256", "malicious", "repack", "family", "malware",
             "apidetected", "similarity", "mas", "y"]
feat_cols = [c for c in focus.columns if c not in meta_cols]

X = focus[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(-1e13)
X = X.loc[:, X.nunique() > 1]          # drop constant columns
y = focus["y"]

report = {
    "dataset": {
        "full_repackaged": int(len(df)),
        "focus_total": int(len(focus)),
        "focus_malware": int((focus["y"] == 1).sum()),
        "focus_benign": int((focus["y"] == 0).sum()),
        "per_family": {f: int(((focus["family"] == f) & (focus["y"] == 1)).sum())
                       for f in FOCUS_FAMILIES},
        "n_features_raw": len(feat_cols),
        "n_features_used": int(X.shape[1]),
        "similarity": {
            f: {
                "mean": round(float(focus.loc[focus["family"] == f, "similarity"].mean()), 4),
                "median": round(float(focus.loc[focus["family"] == f, "similarity"].median()), 4),
                "sd": round(float(focus.loc[focus["family"] == f, "similarity"].std()), 4),
            }
            for f in FOCUS_FAMILIES
        },
    }
}

# -------------------------------------------------------------- MAS baselines
def confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4))

report["mas_on_focus_full"] = confusion(focus["y"], focus["mas"])
report["mas_per_family_full"] = {
    f: confusion(focus.loc[focus["family"] == f, "y"],
                 focus.loc[focus["family"] == f, "mas"])
    for f in FOCUS_FAMILIES
}

# ------------------------------------------------------------------- split
idx_train, idx_test = train_test_split(
    focus.index, test_size=0.3, random_state=SEED, stratify=y
)
X_train, X_test = X.loc[idx_train], X.loc[idx_test]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

report["split"] = {"train": int(len(idx_train)), "test": int(len(idx_test)),
                   "test_malware": int((y_test == 1).sum()),
                   "test_benign": int((y_test == 0).sum())}

# ------------------------------------------------------- feature selection
# Same procedure as the published replication package (top-20 features by Gini
# importance of a Random Forest), but fitted on the TRAINING split only, so the
# selection never sees the test samples. We use a plain forest with default
# settings rather than the tuned one of the replication package, so that the
# selection step has no hyper-parameter of its own to justify.
N_SELECTED = 20
selector = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
selector.fit(X_train, y_train)
gini = pd.Series(selector.feature_importances_, index=X_train.columns)
selected = list(gini.sort_values(ascending=False).head(N_SELECTED).index)
report["selected_features"] = [
    {"feature": f, "gini": round(float(gini[f]), 6)} for f in selected
]
X_train, X_test = X_train[selected], X_test[selected]
report["dataset"]["n_features_selected"] = N_SELECTED

# ------------------------------------------------------------------ models
# Every hyper-parameter is chosen by grid search with stratified k-fold
# cross-validation *on the training split only*, optimising F1. No model sees a
# test sample before its final evaluation, and every model is then applied with
# the standard decision rule (probability >= 0.5 / the natural SVM boundary),
# so that the comparison in the paper is at a common operating point.
grids = {
    "LDA": (
        LDA(solver="eigen"),
        {"shrinkage": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, "auto"], "tol": [1e-4]},
    ),
    "QDA": (
        QDA(store_covariance=True),
        {"reg_param": [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]},
    ),
    "LR": (
        LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED),
        {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "penalty": ["l1", "l2"]},
    ),
    "RF": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 500], "max_depth": [5, 10, None],
         "min_samples_leaf": [1, 3, 10], "max_features": ["sqrt", "log2"]},
    ),
    "MLP": (
        MLPClassifier(solver="adam", max_iter=2000, random_state=SEED),
        {"hidden_layer_sizes": [(10,), (10, 10), (50,), (50, 25)],
         "alpha": [1e-4, 1e-2, 1.0], "activation": ["relu", "tanh"]},
    ),
    "SVM": (
        SVC(kernel="rbf", random_state=SEED),
        {"C": [0.1, 1.0, 5.0, 25.0], "gamma": ["scale", "auto", 0.01, 0.1]},
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 500], "max_depth": [3, 5, 8],
         "learning_rate": [0.05, 0.1, 0.3], "subsample": [0.8, 1.0],
         "colsample_bytree": [0.8, 1.0]},
    ),
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
models, results, preds, best_params = {}, {}, {}, {}
for name, (estimator, grid) in grids.items():
    search = GridSearchCV(estimator, grid, scoring="f1", cv=cv, n_jobs=-1,
                          refit=True)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    models[name] = model
    best_params[name] = {k: (v if not isinstance(v, tuple) else list(v))
                         for k, v in search.best_params_.items()}
    best_params[name]["cv_f1"] = round(float(search.best_score_), 4)

    yhat = model.predict(X_test).astype(int)
    # Score used only for the precision-recall curve, never for the hard label.
    score = (model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba")
             else model.decision_function(X_test))
    preds[name] = yhat
    p, r, f, _ = precision_recall_fscore_support(y_test, yhat, average="binary",
                                                 zero_division=0)
    pr, rc, _ = precision_recall_curve(y_test, score)
    results[name] = confusion(y_test, yhat)
    results[name]["auc"] = round(float(auc(rc, pr)), 4)
    print(f"{name:8s} cvF1={search.best_score_:.3f} | "
          f"P={p:.3f} R={r:.3f} F1={f:.3f} AUC={results[name]['auc']:.3f}")

report["cv"] = {"folds": CV_FOLDS, "scoring": "f1", "best_params": best_params}
report["algorithms"] = results
best = max(results, key=lambda k: (results[k]["f1"], results[k]["auc"]))
report["best_algorithm"] = best

# ------------------------------------------------- best model vs MAS on test
test_meta = focus.loc[idx_test, ["hash", "family", "y", "mas", "similarity"]].copy()
test_meta["flow"] = preds[best]

report["test_comparison"] = {
    "MAS": confusion(test_meta["y"], test_meta["mas"]),
    "DroidXPflow": confusion(test_meta["y"], test_meta["flow"]),
    "Combined": confusion(test_meta["y"], test_meta["mas"] | test_meta["flow"].astype(bool)),
}

mal = test_meta[test_meta["y"] == 1]
report["contribution"] = {
    "TP_flow_only": int(((mal["flow"] == 1) & (~mal["mas"])).sum()),
    "TP_mas_only": int(((mal["flow"] == 0) & (mal["mas"])).sum()),
    "TP_both": int(((mal["flow"] == 1) & (mal["mas"])).sum()),
    "FN_both": int(((mal["flow"] == 0) & (~mal["mas"])).sum()),
    "TP_flow_only_by_family": mal[(mal["flow"] == 1) & (~mal["mas"])]["family"]
                                .value_counts().to_dict(),
    "TP_mas_only_by_family": mal[(mal["flow"] == 0) & (mal["mas"])]["family"]
                               .value_counts().to_dict(),
    "FN_both_by_family": mal[(mal["flow"] == 0) & (~mal["mas"])]["family"]
                            .value_counts().to_dict(),
}

report["per_family_test"] = {}
for f in FOCUS_FAMILIES:
    sub = test_meta[test_meta["family"] == f]
    report["per_family_test"][f] = {
        "samples": int(len(sub)),
        "mas_detected": int(sub["mas"].sum()),
        "mas_rate": round(float(sub["mas"].mean() * 100), 2),
        "flow_detected": int(sub["flow"].sum()),
        "flow_rate": round(float(sub["flow"].mean() * 100), 2),
    }
benign = test_meta[test_meta["y"] == 0]
report["per_family_test"]["benign"] = {
    "samples": int(len(benign)),
    "mas_fp": int(benign["mas"].sum()),
    "flow_fp": int(benign["flow"].sum()),
}

# --------------------------------------------- feature importance (best model)
# NB: this is NOT the Gini importance used for the selection step above. For a
# gradient boosting model it is the gain-based importance of the fitted trees;
# the key below records which one it is so the paper can name it correctly.
imp_model = models[best]
if hasattr(imp_model, "feature_importances_"):
    kind = "gain (XGBoost)" if best == "XGBoost" else "Gini (MDI)"
    imp = pd.Series(imp_model.feature_importances_, index=X_train.columns)
    report["best_importance_kind"] = kind
    report["top_features"] = [
        {"feature": k, "importance": round(float(v), 6)}
        for k, v in imp.sort_values(ascending=False).head(10).items()
    ]

with open(join(OUT, "focus_results.json"), "w") as fh:
    json.dump(report, fh, indent=2)

# Test-split predictions of the selected model, so that focus_similarity.py and
# make_figs.py report on exactly this model rather than refitting their own.
test_meta.to_csv(join(OUT, "focus_predictions.csv"), index=False)
print("\nbest:", best)
print(json.dumps(report["test_comparison"], indent=2))
print(json.dumps(report["per_family_test"], indent=2))
