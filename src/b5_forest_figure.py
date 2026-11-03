"""
Part B, step 5 (figure): coefficient on PromotionSavings across robustness specifications (forest plot).

Reads   part_b_output/b5_tables.xlsx          (written by step B5, sheet "All")
Writes  part_b_output/b5_fig_robustness.png

Run it from main.ipynb:   b5_forest_figure.run(paths)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .config import Paths

BLUE, INK, MUTE, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e3e2de"
PREFERRED = "A6 two-way FE + NecessitySpending (preferred)"

# label, (specification, outcome) for Model 1, (specification, outcome) for Model 2.
# Specification names are the exact strings written by b5_robustness.py.
SPECS = [
    ("Preferred specification", (PREFERRED, "disc_spend"), (PREFERRED, "disc_share")),
    ("Coupon as 0/1 indicator", ("C2 coupon redeemed (0/1)", "disc_spend"), ("C2 coupon redeemed (0/1)", "disc_share")),
    ("Coupon winsorised at p99", ("C3 coupon value winsorised at p99", "disc_spend"), ("C3 coupon value winsorised at p99", "disc_share")),
    ("Outcome: narrow definition", ("D narrow definition ($)", None), ("D narrow definition (pp)", None)),
    ("Outcome: broad definition", ("D broad definition ($)", None), ("D broad definition (pp)", None)),
    ("Outcome: excluding sugary drinks", ("D excluding sugary drinks ($)", None), ("D excluding sugary drinks (pp)", None)),
    ("Outcome: excluding alcohol", ("D excluding alcohol ($)", None), ("D excluding alcohol (pp)", None)),
    (r"Weeks with spend ≥ \$10", ("E1 weeks with total spend >= $10", "disc_spend"), ("E1 weeks with total spend >= $10", "disc_share")),
    ("Weeks with ≤ 10 trips", ("E2 weeks with <= 10 trips", "disc_spend"), ("E2 weeks with <= 10 trips", "disc_share")),
    ("Variables winsorised 1%/99%", ("E3 variables winsorised at 1% / 99%", "disc_spend"), ("E3 variables winsorised at 1% / 99%", "disc_share")),
    ("Households with ≥ 52 weeks", ("E4 households with >= 52 active weeks", "disc_spend"), ("E4 households with >= 52 active weeks", "disc_share")),
    ("All households (no demographics)", ("E5 ALL households >= 10 weeks (no demographics)", "disc_spend"), ("E5 ALL households >= 10 weeks (no demographics)", "disc_share")),
    ("Household-trip panel", ("E6 household-trip (basket) panel", "disc_spend"), ("E6 household-trip (basket) panel", "disc_share")),
    ("Placebo (adds next week's savings)", ("F1 placebo: add NEXT week's savings (should be ~0)", "disc_spend"), ("F1 placebo: add NEXT week's savings (should be ~0)", "disc_share")),
    (r"Income under \$35K", ("H < $35K", "disc_spend"), ("H < $35K", "disc_share")),
    (r"Income \$35-74K", ("H $35-74K", "disc_spend"), ("H $35-74K", "disc_share")),
    (r"Income \$75K+", ("H $75K+", "disc_spend"), ("H $75K+", "disc_share")),
]


def _get(R: pd.DataFrame, spec: str, outcome: str | None, var: str = "promo_nec"):
    t = R[(R["Specification"] == spec) & (R["Variable"] == var)]
    if outcome is not None:
        t = t[t["Outcome"] == outcome]
    if t.empty:
        return None
    r = t.iloc[0]
    return float(r["Coef"]), float(r["SE"])


def run(paths: Paths) -> None:
    table = paths.out_b / "b5_tables.xlsx"
    if not table.exists():
        raise FileNotFoundError(f"{table.name} not found in '{table.parent}'. Run step B5 (b5_robustness) first.")
    R = pd.read_excel(table, sheet_name="All")

    rows = []
    for lab, a, b in SPECS:
        ga, gb = _get(R, *a), _get(R, *b)
        if ga is not None and gb is not None:
            rows.append((lab, ga, gb))

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), sharey=True)
    n = len(rows)
    ys = list(range(n))[::-1]
    panels = [(axes[0], 1, "Model 1: DiscretionarySpending", r"\$ per \$1 of savings"),
              (axes[1], 2, "Model 2: DiscretionaryShare", r"percentage points per \$1 of savings")]
    for ax, idx, title, xl in panels:
        ax.axvline(0, color=INK, lw=1, zorder=1)
        ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
        ax.set_axisbelow(True)
        for y, r in zip(ys, rows):
            c, se = r[idx]
            ax.plot([c - 1.96 * se, c + 1.96 * se], [y, y], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(c, y, "o", ms=8, color=BLUE, mec="white", mew=2, zorder=3)
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.set_xlabel(xl, color=MUTE, fontsize=10)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTE, labelsize=9, length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r[0] for r in rows], color=INK, fontsize=9.5)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    fig.suptitle("Coefficient on PromotionSavings (necessity items) across robustness specifications, with 95% confidence intervals",
                 x=0.01, ha="left", fontsize=11, color=MUTE, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = paths.out_b / "b5_fig_robustness.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved", paths.show(out), f"({n} specifications)")
