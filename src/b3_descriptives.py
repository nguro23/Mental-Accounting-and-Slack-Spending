"""
Part B, step 3: descriptive statistics and figures (Chapter 4.5).

Reads   part_b_output/hw_main.csv                  (main sample, from step B1)
Writes  part_b_output/b3_tables.xlsx               one sheet per table
        part_b_output/b3_*.csv                      the same tables as CSV
        part_b_output/b3_fig_weekly.png            weekly time series
        part_b_output/b3_fig_composition.png       spending composition by household group
        part_b_output/part_b3_report.txt           everything printed

Run it from main.ipynb:   b3_descriptives.run(paths)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Paths
from .utils import (AQUA, BLUE, INK, INK2, ORANGE, PLOT_RC, YELLOW, Reporter, demean, read_panel, style_axes)

LAB = {"disc_spend": "DiscretionarySpending ($)", "disc_share": "DiscretionaryShare (%)",
       "nec_spend": "NecessitySpending ($)", "neutral_spend": "Neutral spending ($)",
       "total_spend": "Total merchandise spend ($)", "promo_nec": "PromotionSavings, necessity items ($)",
       "promo_all": "PromotionSavings, all items ($)", "promo_nec_rate": "Necessity markdown depth (% of shelf value)",
       "coupon": "Coupon value ($)", "coupon_any": "Coupon redeemed (0/1)", "n_baskets": "Shopping trips in the week"}


def run(paths: Paths) -> dict[str, pd.DataFrame]:
    paths.make_dirs()
    rep = Reporter()
    S = read_panel(paths.hw_main)
    S["other_disc_spend"] = S["disc_broad_spend"] - S["disc_spend"]    # extended (borderline) discretionary, not in main
    S["income_grp"] = pd.cut(S["income_k_mid"], [0, 35, 75, 1e6], right=False, labels=["< $35K", "$35-74K", "$75K+"])
    S["size_grp"] = np.where(S["hh_size_num"] >= 3, "3+ persons", S["hh_size_num"].astype(int).astype(str) +
                             np.where(S["hh_size_num"] == 1, " person", " persons"))
    S["kids_grp"] = np.where(S["has_kids"] == 1, "Children (1+)", "None / unknown")

    tables = {}
    tables["Summary_stats"] = _summary_statistics(rep, S)
    tables["Household_characteristics"] = _household_characteristics(rep, S)
    tables["Means_by_group"] = _means_by_group(rep, S)
    tables["Composition"] = _composition(rep, S)
    tables["Correlations"] = _correlations(rep, S)
    tables["Variance_decomposition"] = _variance_decomposition(rep, S)
    _figures(rep, S, tables["Composition"], paths)
    _save(rep, tables, paths)
    return tables


# ---------------------------------------------------------------------------
def _summary_statistics(rep: Reporter, S: pd.DataFrame) -> pd.DataFrame:
    rep.header("1. SUMMARY STATISTICS (household-week level, main sample)")
    rows = []
    for c, lab in LAB.items():
        x = S[c].dropna()
        rows.append([lab, len(x), x.mean(), x.std(), x.min(), x.quantile(.25), x.median(), x.quantile(.75), x.max()])
    T1 = pd.DataFrame(rows, columns=["Variable", "N", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"])
    rep.say(T1.round(2).to_string(index=False))
    rep.say(f"\nHousehold-weeks: {len(S):,}   households: {S['household_key'].nunique()}   weeks: {S['week_no'].nunique()}")
    return T1


def _household_characteristics(rep: Reporter, S: pd.DataFrame) -> pd.DataFrame:
    H = S.drop_duplicates("household_key")
    rep.header(f"2. HOUSEHOLD CHARACTERISTICS ({len(H)} households)")
    blocks = []
    for col, name in [("income_desc", "Income band"), ("size_grp", "Household size"), ("kids_grp", "Children")]:
        if col == "income_desc":
            order = H.sort_values("income_band")["income_desc"].unique()
            vc = H[col].value_counts().reindex(order)
        else:
            vc = H[col].value_counts().sort_index()
        t = pd.DataFrame({"Group": name, "Category": vc.index, "Households": vc.values})
        t["Share (%)"] = 100 * t["Households"] / len(H)
        blocks.append(t)
    T2 = pd.concat(blocks, ignore_index=True)
    rep.say(T2.round(1).to_string(index=False))
    return T2


def _means_by_group(rep: Reporter, S: pd.DataFrame) -> pd.DataFrame:
    rep.header("3. MEANS BY HOUSEHOLD GROUP (household-week means)")

    def group_table(col, title):
        g = S.groupby(col, observed=True)
        t = pd.DataFrame({
            "Households": g["household_key"].nunique(), "Household-weeks": g.size(),
            "Total spend ($)": g["total_spend"].mean(), "Discretionary spend ($)": g["disc_spend"].mean(),
            "Discretionary share (%)": g["disc_share"].mean(), "PromotionSavings, necessity ($)": g["promo_nec"].mean(),
            "Markdown depth (%)": 100 * g["promo_nec"].sum() / g["nec_spend_shelf"].sum(),
            "Weeks with coupon (%)": 100 * g["coupon_any"].mean()})
        t.insert(0, "Group", title)
        return t.reset_index().rename(columns={col: "Category"})

    T3 = pd.concat([group_table("income_grp", "Income"), group_table("size_grp", "Household size"),
                    group_table("kids_grp", "Children")], ignore_index=True)
    rep.say(T3.round(2).to_string(index=False))
    return T3


def _composition(rep: Reporter, S: pd.DataFrame) -> pd.DataFrame:
    rep.header("4. SPENDING COMPOSITION (share of total merchandise spend)")

    def comp(df, label):
        t = df["total_spend"].sum()
        return [label, df["household_key"].nunique(), 100 * df["nec_spend"].sum() / t, 100 * df["disc_spend"].sum() / t,
                100 * df["neutral_spend"].sum() / t, 100 * df["other_disc_spend"].sum() / t]

    crow = [comp(S, "All households")]
    for col in ["income_grp", "kids_grp"]:
        for k, d in S.groupby(col, observed=True):
            crow.append(comp(d, f"{'Income ' if col == 'income_grp' else ''}{k}"))
    T4 = pd.DataFrame(crow, columns=["Group", "Households", "Necessity (%)", "Discretionary, main (%)", "Neutral (%)",
                                     "Discretionary, extended (%)"])
    rep.say(T4.round(1).to_string(index=False))
    return T4


def _correlations(rep: Reporter, S: pd.DataFrame) -> pd.DataFrame:
    rep.header("5. CORRELATIONS (pooled, household-week)")
    cv = ["disc_spend", "disc_share", "nec_spend", "promo_nec", "promo_all", "promo_disc", "coupon", "total_spend"]
    T5 = S[cv].corr().round(2)
    T5.index = T5.columns = ["DiscSpend", "DiscShare", "NecSpend", "Promo(nec)", "Promo(all)", "Promo(disc)", "Coupon", "TotalSpend"]
    rep.say(T5.to_string())
    return T5.reset_index().rename(columns={"index": "Variable"})


def _variance_decomposition(rep: Reporter, S: pd.DataFrame) -> pd.DataFrame:
    rep.header("6. HOW MUCH VARIATION IS WITHIN HOUSEHOLDS? (variance decomposition)")
    rows = []
    for c in ["disc_spend", "disc_share", "nec_spend", "promo_nec", "coupon"]:
        x = S[c].astype(float)
        w_hh = demean(S, [c], ["household_key"])[c]
        r = demean(S, [c], ["household_key", "week_no"], tol=1e-9, max_iter=200)[c]     # two-way: alternating projections
        rows.append([LAB[c], 100 * w_hh.var() / x.var(), 100 * r.var() / x.var()])
    T6 = pd.DataFrame(rows, columns=["Variable", "Within-household share of variance (%)",
                                     "Remaining after household and week effects (%)"])
    rep.say(T6.round(1).to_string(index=False))
    return T6


def _figures(rep: Reporter, S: pd.DataFrame, T4: pd.DataFrame, paths: Paths) -> None:
    rep.header("7. FIGURES")
    with plt.rc_context(PLOT_RC):
        W = S.groupby("week_no").agg(disc_spend=("disc_spend", "mean"), disc_share=("disc_share", "mean"),
                                     promo_nec=("promo_nec", "mean"), hh=("household_key", "nunique")).reset_index()
        fig, axs = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True)
        panels = [("disc_spend", "Mean DiscretionarySpending ($)"), ("disc_share", "Mean DiscretionaryShare (%)"),
                  ("promo_nec", "Mean PromotionSavings, necessity items ($)"), ("hh", "Households observed in the week")]
        for ax, (c, ttl) in zip(axs.ravel(), panels):
            ax.plot(W["week_no"], W[c], color=BLUE, alpha=0.35, linewidth=0.9)
            ax.plot(W["week_no"], W[c].rolling(4, center=True, min_periods=2).mean(), color=BLUE, linewidth=1.8)
            ax.set_title(ttl)
            style_axes(ax)
            if c != "hh":
                ax.set_ylim(bottom=0)
        for ax in axs[1]:
            ax.set_xlabel("Week (1-102)")
        fig.suptitle("Weekly averages across households. Thin line: weekly mean; thick line: 4-week moving average.",
                     x=0.01, ha="left", fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(paths.out_b / "b3_fig_weekly.png", dpi=250)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.4))
        labels = T4["Group"].tolist()[::-1]
        vals = T4[["Necessity (%)", "Discretionary, main (%)", "Neutral (%)", "Discretionary, extended (%)"]].to_numpy()[::-1]
        cols = [BLUE, ORANGE, AQUA, YELLOW]            # stack order follows the validated slot order
        names = ["Necessity", "Discretionary (main)", "Neutral", "Discretionary (extended)"]
        left = np.zeros(len(labels))
        for j in range(4):
            ax.barh(labels, vals[:, j], left=left, color=cols[j], edgecolor="white", linewidth=1.5, height=0.62, label=names[j])
            for i, v in enumerate(vals[:, j]):
                if v >= 4:
                    ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                            color="white" if j == 0 else INK)
            left += vals[:, j]
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of total merchandise spend (%)")
        ax.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5)
        style_axes(ax, grid="x")
        ax.tick_params(axis="y", length=0)
        fig.tight_layout()
        fig.savefig(paths.out_b / "b3_fig_composition.png", dpi=250)
        plt.close(fig)
    rep.say("Saved b3_fig_weekly.png and b3_fig_composition.png")


def _save(rep: Reporter, tables: dict[str, pd.DataFrame], paths: Paths) -> None:
    rep.header("8. SAVE")
    with pd.ExcelWriter(paths.out_b / "b3_tables.xlsx") as xw:
        for k, t in tables.items():
            t.to_excel(xw, sheet_name=k[:31], index=False)
            t.to_csv(paths.out_b / f"b3_{k.lower()}.csv", index=False)
    rep.say("Saved b3_tables.xlsx and one CSV per table in", paths.show(paths.out_b))
    rep.save(paths.out_b / "part_b3_report.txt")
