import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

import os
from os.path import abspath, dirname, join

S = dirname(abspath(__file__))          # focus_*.json live next to this script
# Where the PDFs go. Defaults to this directory; the paper build sets
# FOCUS_FIGS to the directory holding the .tex sources.
OUT = os.environ.get("FOCUS_FIGS", S)

r = json.load(open(f"{S}/focus_results.json"))
c = r["contribution"]

# ---------------------------------------------------------------- Venn of TPs
fig, ax = plt.subplots(figsize=(5.0, 3.4))
v = venn2(subsets=(c["TP_flow_only"], c["TP_mas_only"], c["TP_both"]),
          set_labels=("DroidXPflow", "MAS approach"),
          set_colors=("#4C9F70", "#3B6EA5"), alpha=0.65, ax=ax)
for t in v.set_labels:
    if t:
        t.set_fontsize(11)
for t in v.subset_labels:
    if t:
        t.set_fontsize(11)
        t.set_fontweight("bold")
ax.set_title(f"True positives ({c['TP_flow_only']+c['TP_mas_only']+c['TP_both']} of "
             f"{r['split']['test_malware']} malware samples;\n"
             f"{c['FN_both']} missed by both approaches)", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/vennFocusTP.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- detection rate by family
sim = json.load(open(f"{S}/focus_similarity.json"))["similarity_bands"]
bands = [b for b in sim if b["samples"] >= 20]
labels = [b["band"] for b in bands]
mas = [b["mas_rate"] for b in bands]
flow = [b["flow_rate"] for b in bands]

fig, ax = plt.subplots(figsize=(5.2, 3.0))
x = range(len(labels))
w = 0.36
ax.bar([i - w / 2 for i in x], mas, w, label="MAS approach", color="#3B6EA5")
ax.bar([i + w / 2 for i in x], flow, w, label="DroidXPflow", color="#4C9F70")
for i, (m, f) in enumerate(zip(mas, flow)):
    ax.text(i - w / 2, m + 1.5, f"{m:.0f}", ha="center", fontsize=8)
    ax.text(i + w / 2, f + 1.5, f"{f:.0f}", ha="center", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 105)
ax.set_xlabel("Similarity Score band", fontsize=9)
ax.set_ylabel("Malware correctly detected (\\%)", fontsize=9)
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/similarityBands.pdf", bbox_inches="tight")
plt.close(fig)
print("figures written")
