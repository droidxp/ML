"""Check a local run against the numbers published in the paper.

Every value below is transcribed from the tables and findings of the extended
paper, grouped by where it appears. Run focus_study.py and focus_similarity.py
first, then:

    python3 verify.py

Exits 0 if every value matches, 1 otherwise, printing the offending ones.
Integers must match exactly; rounded figures are compared at the precision the
paper prints them with.
"""
import json
import sys
from os.path import abspath, dirname, join

HERE = dirname(abspath(__file__))

# (group, description, file, dotted path, expected, tolerance)
# tolerance 0 => exact; otherwise abs(actual - expected) <= tolerance
R, S = "focus_results.json", "focus_similarity.json"

CHECKS = [
    # ---------------------------------------------------------- Section 6.1
    ("Sec. 6.1  FocusDS", "distinct repackaged apps in LargeDS", R, "dataset.full_repackaged", 4067, 0),
    ("Sec. 6.1  FocusDS", "FocusDS size", R, "dataset.focus_total", 2722, 0),
    ("Sec. 6.1  FocusDS", "malware samples", R, "dataset.focus_malware", 1541, 0),
    ("Sec. 6.1  FocusDS", "non-malware samples", R, "dataset.focus_benign", 1181, 0),
    ("Sec. 6.1  FocusDS", "gappusin samples", R, "dataset.per_family.gappusin", 1334, 0),
    ("Sec. 6.1  FocusDS", "revmob samples", R, "dataset.per_family.revmob", 207, 0),
    ("Sec. 6.1  FocusDS", "MAS true positives on FocusDS", R, "mas_on_focus_full.TP", 279, 0),
    ("Sec. 6.1  FocusDS", "MAS false positives on FocusDS", R, "mas_on_focus_full.FP", 220, 0),
    ("Sec. 6.1  FocusDS", "MAS false negatives on FocusDS", R, "mas_on_focus_full.FN", 1262, 0),
    ("Sec. 6.1  FocusDS", "MAS F1 on FocusDS", R, "mas_on_focus_full.f1", 0.27, 0.005),
    ("Sec. 6.1  FocusDS", "MAS gappusin TP (dedup.)", R, "mas_per_family_full.gappusin.TP", 164, 0),
    ("Sec. 6.1  FocusDS", "MAS gappusin FN", R, "mas_per_family_full.gappusin.FN", 1170, 0),
    ("Sec. 6.1  FocusDS", "MAS revmob TP", R, "mas_per_family_full.revmob.TP", 115, 0),
    ("Sec. 6.1  FocusDS", "MAS revmob FN", R, "mas_per_family_full.revmob.FN", 92, 0),

    # ------------------------------------------------ Section 6.1 (protocol)
    ("Sec. 6.1  Protocol", "features extracted per app", R, "dataset.n_features_raw", 1596, 0),
    ("Sec. 6.1  Protocol", "non-constant features", R, "dataset.n_features_used", 1010, 0),
    ("Sec. 6.1  Protocol", "features selected", R, "dataset.n_features_selected", 20, 0),
    ("Sec. 6.1  Protocol", "training split size", R, "split.train", 1905, 0),
    ("Sec. 6.1  Protocol", "test split size", R, "split.test", 817, 0),
    ("Sec. 6.1  Protocol", "test malware", R, "split.test_malware", 463, 0),
    ("Sec. 6.1  Protocol", "test non-malware", R, "split.test_benign", 354, 0),

    # --------------------------------------------------- Table 8 (CV tuning)
    ("Table 8   Hyper-params", "LDA cross-validated F1", R, "cv.best_params.LDA.cv_f1", 0.742, 0.0005),
    ("Table 8   Hyper-params", "QDA cross-validated F1", R, "cv.best_params.QDA.cv_f1", 0.744, 0.0005),
    ("Table 8   Hyper-params", "LR cross-validated F1", R, "cv.best_params.LR.cv_f1", 0.742, 0.0005),
    ("Table 8   Hyper-params", "RF cross-validated F1", R, "cv.best_params.RF.cv_f1", 0.861, 0.0005),
    ("Table 8   Hyper-params", "MLP cross-validated F1", R, "cv.best_params.MLP.cv_f1", 0.727, 0.0005),
    ("Table 8   Hyper-params", "SVM cross-validated F1", R, "cv.best_params.SVM.cv_f1", 0.728, 0.0005),
    ("Table 8   Hyper-params", "XGBoost cross-validated F1", R, "cv.best_params.XGBoost.cv_f1", 0.873, 0.0005),

    # ------------------------------------------------- Table 9 (algorithms)
    *[c for alg, tp, fp, fn, p, r, f1, a in [
        ("LDA", 384, 227, 79, 0.63, 0.83, 0.72, 0.72),
        ("QDA", 398, 235, 65, 0.63, 0.86, 0.73, 0.72),
        ("LR", 387, 225, 76, 0.63, 0.84, 0.72, 0.72),
        ("RF", 393, 36, 70, 0.92, 0.85, 0.88, 0.94),
        ("MLP", 460, 352, 3, 0.57, 0.99, 0.72, 0.68),
        ("SVM", 430, 307, 33, 0.58, 0.93, 0.72, 0.61),
        ("XGBoost", 405, 48, 58, 0.89, 0.87, 0.88, 0.95),
    ] for c in [
        ("Table 9   Algorithms", f"{alg} TP", R, f"algorithms.{alg}.TP", tp, 0),
        ("Table 9   Algorithms", f"{alg} FP", R, f"algorithms.{alg}.FP", fp, 0),
        ("Table 9   Algorithms", f"{alg} FN", R, f"algorithms.{alg}.FN", fn, 0),
        ("Table 9   Algorithms", f"{alg} precision", R, f"algorithms.{alg}.precision", p, 0.005),
        ("Table 9   Algorithms", f"{alg} recall", R, f"algorithms.{alg}.recall", r, 0.005),
        ("Table 9   Algorithms", f"{alg} F1", R, f"algorithms.{alg}.f1", f1, 0.005),
        ("Table 9   Algorithms", f"{alg} AUC", R, f"algorithms.{alg}.auc", a, 0.005),
    ]],
    ("Table 9   Algorithms", "selected algorithm", R, "best_algorithm", "XGBoost", 0),

    # ------------------------------------------------ Table 10 (importances)
    ("Table 10  Importances", "top feature", R, "top_features.0.feature", "bwd_byts_b_avg_median_443", 0),
    ("Table 10  Importances", "top feature importance", R, "top_features.0.importance", 0.1908, 0.0001),
    ("Table 10  Importances", "2nd feature importance", R, "top_features.1.importance", 0.0775, 0.0001),
    ("Table 10  Importances", "3rd feature importance", R, "top_features.2.importance", 0.0733, 0.0001),

    # ------------------------------------------------- Table 11 (comparison)
    ("Table 11  MAS vs flow", "MAS TP (test)", R, "test_comparison.MAS.TP", 87, 0),
    ("Table 11  MAS vs flow", "MAS FP (test)", R, "test_comparison.MAS.FP", 78, 0),
    ("Table 11  MAS vs flow", "MAS FN (test)", R, "test_comparison.MAS.FN", 376, 0),
    ("Table 11  MAS vs flow", "MAS F1 (test)", R, "test_comparison.MAS.f1", 0.28, 0.005),
    ("Table 11  MAS vs flow", "DroidXPflow TP", R, "test_comparison.DroidXPflow.TP", 405, 0),
    ("Table 11  MAS vs flow", "DroidXPflow FP", R, "test_comparison.DroidXPflow.FP", 48, 0),
    ("Table 11  MAS vs flow", "DroidXPflow FN", R, "test_comparison.DroidXPflow.FN", 58, 0),
    ("Table 11  MAS vs flow", "DroidXPflow F1", R, "test_comparison.DroidXPflow.f1", 0.88, 0.005),
    ("Table 11  MAS vs flow", "Combined TP", R, "test_comparison.Combined.TP", 420, 0),
    ("Table 11  MAS vs flow", "Combined FP", R, "test_comparison.Combined.FP", 117, 0),
    ("Table 11  MAS vs flow", "Combined F1", R, "test_comparison.Combined.f1", 0.84, 0.005),

    # --------------------------------------------------- Table 12 (families)
    ("Table 12  Per family", "gappusin test samples", R, "per_family_test.gappusin.samples", 405, 0),
    ("Table 12  Per family", "gappusin MAS rate", R, "per_family_test.gappusin.mas_rate", 13.33, 0.01),
    ("Table 12  Per family", "gappusin DroidXPflow rate", R, "per_family_test.gappusin.flow_rate", 87.41, 0.01),
    ("Table 12  Per family", "revmob test samples", R, "per_family_test.revmob.samples", 58, 0),
    ("Table 12  Per family", "revmob MAS rate", R, "per_family_test.revmob.mas_rate", 56.90, 0.01),
    ("Table 12  Per family", "revmob DroidXPflow rate", R, "per_family_test.revmob.flow_rate", 87.93, 0.01),
    ("Table 12  Per family", "non-malware MAS false positives", R, "per_family_test.benign.mas_fp", 78, 0),
    ("Table 12  Per family", "non-malware flow false positives", R, "per_family_test.benign.flow_fp", 48, 0),

    # ------------------------------------------- Figure 7 (Venn, Finding 7)
    ("Figure 7  Venn", "true positives only DroidXPflow", R, "contribution.TP_flow_only", 333, 0),
    ("Figure 7  Venn", "true positives only MAS", R, "contribution.TP_mas_only", 15, 0),
    ("Figure 7  Venn", "true positives by both", R, "contribution.TP_both", 72, 0),
    ("Figure 7  Venn", "missed by both", R, "contribution.FN_both", 43, 0),

    # --------------------------------- Table 13 + Finding 8 (Similarity, RQ5)
    ("Table 13  Similarity", "[0.75,0.95) samples", S, "similarity_bands.3.samples", 104, 0),
    ("Table 13  Similarity", "[0.75,0.95) MAS rate", S, "similarity_bands.3.mas_rate", 23.08, 0.01),
    ("Table 13  Similarity", "[0.75,0.95) flow rate", S, "similarity_bands.3.flow_rate", 85.58, 0.01),
    ("Table 13  Similarity", "[0.95,1.00] samples", S, "similarity_bands.4.samples", 341, 0),
    ("Table 13  Similarity", "[0.95,1.00] MAS rate", S, "similarity_bands.4.mas_rate", 14.66, 0.01),
    ("Table 13  Similarity", "[0.95,1.00] flow rate", S, "similarity_bands.4.flow_rate", 87.98, 0.01),
    ("Table 13  Similarity", "Spearman rho, MAS", S, "spearman_mas.rho", -0.15, 0.005),
    ("Table 13  Similarity", "Spearman p, MAS (significant)", S, "spearman_mas.p_value", 0.001, 0.0005),
    ("Table 13  Similarity", "Spearman rho, DroidXPflow", S, "spearman_flow.rho", 0.04, 0.005),
    ("Table 13  Similarity", "Spearman p, DroidXPflow (n.s.)", S, "spearman_flow.p_value", 0.42, 0.005),
]


def dig(blob, path):
    cur = blob
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def main():
    data = {}
    for fname in (R, S):
        try:
            with open(join(HERE, fname)) as fh:
                data[fname] = json.load(fh)
        except FileNotFoundError:
            sys.exit(f"{fname} not found. Run focus_study.py and "
                     f"focus_similarity.py first.")

    failures, groups = [], {}
    for group, desc, fname, path, expected, tol in CHECKS:
        groups.setdefault(group, [0, 0])
        try:
            actual = dig(data[fname], path)
        except (KeyError, IndexError):
            failures.append((group, desc, path, expected, "<missing>"))
            groups[group][1] += 1
            continue
        if isinstance(expected, str):
            ok = actual == expected
        elif tol == 0:
            ok = actual == expected
        else:
            ok = abs(float(actual) - float(expected)) <= tol
        if ok:
            groups[group][0] += 1
        else:
            groups[group][1] += 1
            failures.append((group, desc, path, expected, actual))

    width = max(len(g) for g in groups)
    print("Checking this run against the numbers published in the paper\n")
    for group, (ok, bad) in groups.items():
        status = "OK  " if bad == 0 else "FAIL"
        print(f"  [{status}] {group:<{width}}  {ok}/{ok + bad} values match")

    total_ok = sum(o for o, _ in groups.values())
    total = sum(o + b for o, b in groups.values())
    print()
    if failures:
        print(f"{len(failures)} of {total} values differ:\n")
        for group, desc, path, expected, actual in failures:
            print(f"  {group} | {desc}")
            print(f"      paper: {expected!r}   this run: {actual!r}   ({path})")
        print("\nIf you changed the library versions, see the Environment "
              "section of README.md: the pipeline is deterministic for a given\n"
              "set of versions, but estimators change across releases.")
        return 1
    print(f"All {total_ok} values reproduce the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
