"""Report figures (static PDF) for the election experiments.
Palette: reference categorical slots 1-3 (blue / orange / aqua), validated all-pairs for <=3 series.
Identity is never colour-alone: legend + distinct markers + direct labels on the last point."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import score_politics as S
from common import ROOT

OUT = ROOT / "report" / "figs"; OUT.mkdir(parents=True, exist_ok=True)
C = {"none": "#2a78d6", "collapse": "#eb6834", "annotate": "#1baf7a", "note": "#52514e"}
MK = {"none": "o", "collapse": "s", "annotate": "^", "note": "D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def ci(v):
    return v[0], v[0] - v[1], v[2] - v[0]


def fig_k(res):
    ks = [1, 3, 10, 30]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (m, lab) in zip(axs, [("invite", "P(invite to debate)"), ("momentum", "Grassroots momentum (0–3)"),
                                   ("voice_infl", "Voice inflation (reported − ref.)")]):
        for key, name in [("none|-", "none"), ("annotate|tea", "annotate"), ("collapse|tea", "collapse")]:
            pts = [res["k"].get(f"k={k}|{key}", {}).get(m) for k in ks]
            xs = [k for k, p in zip(ks, pts) if p]; ys = [ci(p) for p in pts if p]
            if not ys: continue
            ax.errorbar(xs, [y[0] for y in ys], yerr=[[y[1] for y in ys], [y[2] for y in ys]], color=C[name], marker=MK[name],
                        ms=5, lw=2, capsize=2, label={"none": "No defense", "annotate": "Annotate @tea", "collapse": "Collapse @tea"}[name])
        ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks); ax.minorticks_off()
        ax.set_title(lab, fontsize=9, loc="left", color=INK)
    fig.supxlabel("Accounts carrying the same 30 pitch texts (k)", fontsize=8, color=MUTED)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "k_ablation.pdf"); plt.close(fig)


def fig_nat(res, note):
    groups = [("A", "No swarm"), ("S4", "S4\norchestrated"), ("S5", "S5\nself-seeded")]
    vars_ = [("none|-", "No defense", "none"), ("note", "Warning note only", "note"),
             ("annotate|b5", "Annotate @5%", "annotate"), ("collapse|b5", "Collapse @5%", "collapse")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, (m, lab) in zip(axs, [("invite", "P(invite to debate)"), ("poll", "P(include in poll)")]):
        w = 0.19
        for j, (key, name, col) in enumerate(vars_):
            for i, (g, _) in enumerate(groups):
                v = note.get(f"{g}|note-only", {}).get(m) if key == "note" else res["nat"].get(f"{g}|{key}", {}).get(m)
                if not v: continue
                e = ci(v)
                ax.bar(i + (j - 1.5) * w, e[0], w - 0.03, color=C[col], label=name if i == 0 else None)
                ax.errorbar(i + (j - 1.5) * w, e[0], yerr=[[e[1]], [e[2]]], color=INK, lw=0.8, capsize=1.5)
        ax.set_xticks(range(3)); ax.set_xticklabels([g[1] for g in groups]); ax.set_ylim(0, 1.05)
        ax.set_title(lab, fontsize=9, loc="left", color=INK); ax.grid(axis="x", visible=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "natural.pdf"); plt.close(fig)


def fig_oc2(res):
    rows = [("annotate|noflag", "Note only (0 flagged)"), ("collapse|b5", "Collapse @5% (in-domain)"), ("collapse|tea", "Collapse @tea (transferred)"),
            ("annotate|b5", "Annotate @5%"), ("annotate|tea", "Annotate @tea")]
    cands = [("蔣萬安", "Chiang Wan-an (incumbent)", "#2a78d6", "o"), ("沈伯洋", "Puma Shen (challenger)", "#eb6834", "s")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for k, (cand, nm, col, mk) in enumerate(cands):
        for i, (key, _) in enumerate(rows):
            r = res["oc2"].get(key, {}).get(cand, {})
            y = i + (k - 0.5) * 0.32
            axs[0].barh(y, r.get("flagged_supportive_of_24", 0), 0.3, color=col, label=nm if i == 0 else None)
            v = r.get("d_voices")
            if v:
                e = ci(v)
                axs[1].errorbar(e[0], y, xerr=[[e[1]], [e[2]]], color=col, marker=mk, ms=5, lw=2, capsize=2)
    axs[0].set_title("Detector: supportive msgs flagged (/24)", fontsize=9, loc="left", color=INK)
    axs[1].set_title("Brief: Δ independent voices", fontsize=9, loc="left", color=INK)
    axs[1].axvline(0, color=MUTED, lw=0.8)
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([r[1] for r in rows]); axs[0].invert_yaxis()
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "overcorrection.pdf"); plt.close(fig)


def fig_b1():
    """Camp alert rates vs overall organic alert rate: in-domain budget sweep (steps) and the transferred tea point."""
    b1 = json.load(open(S.DATA / "politics" / "b1_matched_rate.json"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.05), sharey=True)
    for ax, win, ttl in zip(axs, ("dev", "heldout"), ("Dev window", "Held-out window")):
        rows = sorted([r for r in b1[win] if r["op"] != "tea"], key=lambda r: (r["alert"], r["op"]))
        tea = next(r for r in b1[win] if r["op"] == "tea")
        for key, nm, col, mk in (("chiang", "Incumbent's supporters", "#2a78d6", "o"), ("shen", "Challenger's supporters", "#eb6834", "s")):
            ax.plot([r["alert"] * 100 for r in rows], [r[key] * 100 for r in rows], color=col, marker=mk, ms=4, ls="none", label=nm + " (in-domain budgets)")
            ax.plot(tea["alert"] * 100, tea[key] * 100, color=col, marker="*", ms=12, ls="none", mec=INK, mew=0.6, label=nm + " (transferred 'tea')")
        ax.set_title(ttl, fontsize=9, loc="left", color=INK); ax.set_xlabel("Overall organic alert rate (%)", fontsize=8, color=MUTED)
    axs[0].set_ylabel("Supporters flagged (%)", fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.82)); fig.savefig(OUT / "b1_matched.pdf"); plt.close(fig)


if __name__ == "__main__":
    res = json.load(open(S.DATA / "politics" / "results.json"))
    fig_k(res)
    fig_nat(res, res.get("note", {}))
    if "oc2" in res: fig_oc2(res)
    fig_b1()
    print("figs ->", OUT)
