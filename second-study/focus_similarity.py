"""Addendum: does DroidXPflow also remove the Similarity-Score blind spot that
the ECOOP study reported (Finding 3)?

Reads the test-split predictions that focus_study.py wrote, so that this
analysis reports on exactly the model selected there -- same subset, same
split, same hyper-parameters, same decision rule.

Run focus_study.py first.
"""
import json
from os.path import abspath, dirname, join

import pandas as pd
from scipy.stats import spearmanr

HERE = dirname(abspath(__file__))

t = pd.read_csv(join(HERE, "focus_predictions.csv"))
t["mas"] = t["mas"].astype(bool)

bands = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.95), (0.95, 1.01)]
out = {"similarity_bands": [],
       "note": "malware samples of the focus subset, test split"}
mal = t[t.y == 1]
for lo, hi in bands:
    s = mal[(mal.similarity >= lo) & (mal.similarity < hi)]
    if len(s) == 0:
        continue
    out["similarity_bands"].append({
        "band": f"[{lo:.2f}, {hi:.2f})" if hi <= 1.0 else f"[{lo:.2f}, 1.00]",
        "samples": int(len(s)),
        "mas_detected": int(s.mas.sum()),
        "mas_rate": round(float(s.mas.mean() * 100), 2),
        "flow_detected": int(s.flow.sum()),
        "flow_rate": round(float(s.flow.mean() * 100), 2),
    })

# Spearman between similarity and correctness, for each approach
for name, col in (("mas", "mas"), ("flow", "flow")):
    rho, p = spearmanr(mal["similarity"], mal[col].astype(int))
    out[f"spearman_{name}"] = {"rho": round(float(rho), 4), "p_value": float(p)}

print(json.dumps(out, indent=2))
with open(join(HERE, "focus_similarity.json"), "w") as fh:
    json.dump(out, fh, indent=2)
