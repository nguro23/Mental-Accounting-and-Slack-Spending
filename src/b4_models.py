"""
Part B, step 4: staged regression models, hypothesis tests and figures (Chapters 4.6 and 4.7).

Reads   part_b_output/hw_main.csv                  (main sample, from step B1)
Writes  part_b_output/b4_tables.xlsx               Model 1, Model 2, hypothesis decisions, inference check, effect sizes
        part_b_output/b4_*.csv                      the same tables as CSV
        part_b_output/b4_fig_stability.png         beta on PromotionSavings across the four columns, with 95% CIs
        part_b_output/b4_fig_binned.png            binned partial-regression plot of the preferred (two-way FE) model
        part_b_output/part_b4_report.txt           everything printed

Run it from main.ipynb:   b4_models.run(paths)

Models (Chapter 3). Y = disc_spend (Model 1, dollars)  or  disc_share (Model 2, percentage points)
    (1) Pooled OLS, PromotionSavings + Coupon only                        [raw association, extra baseline]
    (a) Pooled OLS + NecessitySpending + demographics + week effects
    (b) Mundlak correlated random effects: (a) + household means of the three time-varying regressors
    (c) Two-way fixed effects: household + week effects, NecessitySpending (PREFERRED)
Demographics are time-invariant, so they are absorbed in (c) and appear only in (a) and (b).
All standard errors are clustered by household (two-way household x week clustering is reported as a check).
Hypotheses are tested one-sided at 5% in the direction of the alternative (positive effect).
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import Paths
from .utils import BLUE, GRID, INK2, ORANGE, PLOT_RC, Reporter, demean, quiet_warnings, read_panel, star

XV = ["promo_nec", "coupon", "nec_spend"]
DEMO = ["income_10k", "hh_size_num", "has_kids"]
ROWS = ["promo_nec", "coupon", "nec_spend", "income_10k", "hh_size_num", "has_kids"]
LAB = {"promo_nec": "PromotionSavings (necessity items, $)", "coupon": "Coupon value ($)",
       "nec_spend": "NecessitySpending ($)", "income_10k": "Income (per $10k, band midpoint)",
       "hh_size_num": "Household size (persons)", "has_kids": "Has children (0/1)"}
OUTCOMES = {"Model 1: DiscretionarySpending ($)": "disc_spend", "Model 2: DiscretionaryShare (pp)": "disc_share"}
COLS = ["(1) Pooled, no controls", "(a) Pooled + controls", "(b) Mundlak CRE", "(c) Two-way FE"]


@dataclass
class Data:
    """The main panel plus everything the four estimators need (within-transformed data, dummies, cluster codes)."""
    S: pd.DataFrame
    DM: pd.DataFrame      # two-way within-transformed outcomes and regressors
    Wd: pd.DataFrame      # week dummies
    Xbar: pd.DataFrame    # household means of the time-varying regressors (Mundlak)
    hh: np.ndarray        # household cluster codes
    wk: np.ndarray        # week cluster codes
    n: int
    N: int
    T: int

    @property
    def groups2(self) -> np.ndarray:
        return np.column_stack([self.hh, self.wk])


@quiet_warnings()
def run(paths: Paths) -> dict[str, pd.DataFrame]:
    paths.make_dirs()
    rep = Reporter()

    S = read_panel(paths.hw_main)
    S["income_10k"] = S["income_k_mid"] / 10.0
    d = Data(S=S,
             DM=demean(S, ["disc_spend", "disc_share"] + XV, ["household_key", "week_no"]),
             Wd=pd.get_dummies(S["week_no"], prefix="w", drop_first=True, dtype=float),
             Xbar=S.groupby("household_key")[XV].transform("mean").add_suffix("_bar"),
             hh=pd.factorize(S["household_key"])[0], wk=pd.factorize(S["week_no"])[0],
             n=len(S), N=S["household_key"].nunique(), T=S["week_no"].nunique())
    rep.say(f"Main sample: {d.n:,} household-weeks, {d.N} households, {d.T} weeks")

    res, tabs = {}, {}
    for title, ycol in OUTCOMES.items():
        rep.header(f"{title.upper()}  -  coefficients with household-clustered SEs in parentheses (* p<0.10, ** p<0.05, *** p<0.01)")
        res[ycol] = _fit_stages(d, ycol)
        tabs[ycol] = _model_table(d, res[ycol])
        rep.say(tabs[ycol].to_string(index=False))
        rep.say(f"Mean of the dependent variable: {S[ycol].mean():.3f}")

    decisions = _hypothesis_tests(rep, res)
    inference = _inference_check(rep, res)
    effects = _effect_sizes(rep, d, res)
    _figures(rep, d, res, paths)

    rep.header("SAVE")
    tables = {"Model1": tabs["disc_spend"], "Model2": tabs["disc_share"], "Hypotheses": decisions,
              "Inference_check": inference, "Effect_sizes": effects}
    with pd.ExcelWriter(paths.out_b / "b4_tables.xlsx") as xw:
        for name, t in tables.items():
            t.to_excel(xw, sheet_name=name, index=False)
            t.to_csv(paths.out_b / f"b4_{name.lower()}.csv", index=False)
    rep.say("Saved b4_tables.xlsx and one CSV per table in", paths.show(paths.out_b))
    rep.save(paths.out_b / "part_b4_report.txt")
    return tables


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def _clustered(y, Z, groups):
    return sm.OLS(y, Z).fit(cov_type="cluster", cov_kwds={"groups": groups})


def _fit_stages(d: Data, ycol: str) -> dict:
    """Columns (1), (a), (b), (c) for one outcome, all with household-clustered SEs."""
    S = d.S
    y = S[ycol].to_numpy()
    res = {}
    Z1 = sm.add_constant(S[["promo_nec", "coupon"]])
    res[COLS[0]] = dict(r=_clustered(y, Z1, d.hh), n=d.n, r2=None, label="R-squared")
    Za = sm.add_constant(pd.concat([S[XV + DEMO], d.Wd], axis=1))
    res[COLS[1]] = dict(r=_clustered(y, Za, d.hh))
    Zb = sm.add_constant(pd.concat([S[XV + DEMO], d.Xbar, d.Wd], axis=1))
    res[COLS[2]] = dict(r=_clustered(y, Zb, d.hh))
    Yc, Zc = d.DM[ycol], d.DM[XV]
    rc = _clustered(Yc.to_numpy(), Zc, d.hh)
    rc2 = _clustered(Yc.to_numpy(), Zc, d.groups2)
    rcw = _clustered(Yc.to_numpy(), Zc, d.wk)
    rss = float((rc.resid ** 2).sum())
    res[COLS[3]] = dict(r=rc, r2tw=rc2, rwk=rcw, within_r2=1 - rss / float((Yc ** 2).sum()))
    for k in COLS[:3]:
        res[k]["r2v"] = float(res[k]["r"].rsquared)
    return res


def _model_table(d: Data, res: dict) -> pd.DataFrame:
    out = []
    for v in ROWS:
        c1, c2 = [LAB[v]], [""]
        for k in COLS:
            r = res[k]["r"]
            if v in r.params.index if hasattr(r.params, "index") else False:
                c1.append(f"{r.params[v]:.3f}{star(r.pvalues[v])}")
                c2.append(f"({r.bse[v]:.3f})")
            else:
                c1.append("")
                c2.append("")
        out += [c1, c2]
    out += [["Observations"] + [f"{d.n:,}"] * 4, ["Households"] + [f"{d.N}"] * 4,
            ["R-squared (columns 1, a, b); within R-squared (column c)"] +
            [f"{res[k]['r2v']:.3f}" for k in COLS[:3]] + [f"{res[COLS[3]]['within_r2']:.3f}"],
            ["Week fixed effects", "No", "Yes", "Yes", "Yes"], ["Household fixed effects", "No", "No", "No", "Yes"],
            ["Demographic controls", "No", "Yes", "Yes", "absorbed"],
            ["Household means of regressors (Mundlak)", "No", "No", "Yes", "n/a"]]
    return pd.DataFrame(out, columns=["Variable"] + COLS)


def _p_one(b: float, p2: float) -> float:
    """One-sided p-value for H1: beta > 0, from the two-sided p-value."""
    return p2 / 2 if b > 0 else 1 - p2 / 2


def _get(res: dict, col: str, v: str, which: str = "r"):
    r = res[col][which]
    return float(r.params[v]), float(r.bse[v]), float(r.pvalues[v])


# ---------------------------------------------------------------------------
# Hypothesis tests, inference check, effect sizes
# ---------------------------------------------------------------------------
def _hypothesis_tests(rep: Reporter, res: dict) -> pd.DataFrame:
    rep.header("HYPOTHESIS TESTS (one-sided, 5%; direction = positive effect)")
    dec = []

    def add(h, desc, b, se, p2, p1, p1_tw, note=""):
        reject = (b > 0) and (p1 < 0.05)
        robust = "Yes" if ((b > 0) and (p1_tw < 0.05)) == reject else "No (differs)"
        dec.append(dict(Hypothesis=h, Test=desc, Coefficient=round(b, 4), SE=round(se, 4), p_two_sided=round(p2, 4),
                        p_one_sided=round(p1, 4), Decision="Reject H0" if reject else "Fail to reject H0",
                        p_one_sided_two_way_cluster=round(p1_tw, 4), Same_decision_two_way_clustering=robust, Note=note))

    r1, r2 = res["disc_spend"], res["disc_share"]
    for h, rr, v, desc in [("H1", r1, "promo_nec", "Model 1 (c): beta on PromotionSavings > 0"),
                           ("H2 (savings)", r2, "promo_nec", "Model 2 (c): beta on PromotionSavings > 0"),
                           ("H2 (coupon)", r2, "coupon", "Model 2 (c): beta on Coupon > 0")]:
        b, se, p2 = _get(rr, COLS[3], v)
        _, _, p2t = _get(rr, COLS[3], v, "r2tw")
        add(h, desc, b, se, p2, _p_one(b, p2), _p_one(b, p2t))

    w = res["disc_share"][COLS[3]]["r"].wald_test(np.array([[1, 0, 0], [0, 1, 0]]), use_f=False, scalar=True)
    rep.say(f"H2 joint Wald test, Model 2 (c): beta(PromotionSavings) = beta(Coupon) = 0:  chi2(2) = "
            f"{float(np.squeeze(w.statistic)):.2f},  p = {float(np.squeeze(w.pvalue)):.4f}")

    for (h, rr, lab) in [("H3 (Model 1)", r1, "Model 1"), ("H3 (Model 2, supplementary)", r2, "Model 2")]:
        for col in COLS[1:]:
            b, se, p2 = _get(rr, col, "promo_nec")
            if col == COLS[3]:
                _, _, p2t = _get(rr, col, "promo_nec", "r2tw")
            else:
                p2t = p2
            add(h, f"{lab}, {col}: beta on PromotionSavings > 0", b, se, p2, _p_one(b, p2), _p_one(b, p2t))
    D = pd.DataFrame(dec)
    rep.say(D.drop(columns=["Note"]).to_string(index=False))

    for h in ["H3 (Model 1)", "H3 (Model 2, supplementary)"]:
        sub = D[D.Hypothesis == h]
        persists = bool((sub["Decision"] == "Reject H0").all())
        rep.say(f"\n{h}: effect positive and significant in ALL of (a), (b), (c)?  {persists}  "
                f"-> {'reject H0 (effect persists)' if persists else 'fail to reject H0 / effect does not persist'}")
    return D


def _inference_check(rep: Reporter, res: dict) -> pd.DataFrame:
    rep.header("INFERENCE CHECK (column c): household-clustered vs week-clustered vs two-way clustered SEs")
    inf = []
    for title, ycol in OUTCOMES.items():
        for v in XV:
            b, se_h, p_h = _get(res[ycol], COLS[3], v)
            _, se_t, p_t = _get(res[ycol], COLS[3], v, "r2tw")
            _, se_w, p_w = _get(res[ycol], COLS[3], v, "rwk")
            inf.append([title.split(":")[0], LAB[v], b, se_h, p_h, se_w, p_w, se_t, p_t])
    INF = pd.DataFrame(inf, columns=["Model", "Variable", "Coef", "SE (household)", "p (household)", "SE (week)",
                                     "p (week)", "SE (two-way)", "p (two-way)"])
    rep.say(INF.round(4).to_string(index=False))
    return INF


def _effect_sizes(rep: Reporter, d: Data, res: dict) -> pd.DataFrame:
    rep.header("EFFECT SIZES AND INTERPRETATION AIDS (column c)")
    S = d.S
    eff = []
    bench = float(S["disc_spend"].sum() / S["total_spend"].sum())
    for title, ycol in OUTCOMES.items():
        b, se, p2 = _get(res[ycol], COLS[3], "promo_nec")
        sd_w = float(d.DM["promo_nec"].std())
        ybar, xbar = float(S[ycol].mean()), float(S["promo_nec"].mean())
        unit = "$" if ycol == "disc_spend" else "pp"
        model = title.split(":")[0]
        eff.append([model, f"beta on PromotionSavings ({unit} per $1)", b, ""])
        eff.append([model, "95% CI (household-clustered)", f"[{b - 1.96 * se:.3f}, {b + 1.96 * se:.3f}]", ""])
        eff.append([model, "within-household SD of PromotionSavings ($)", sd_w, ""])
        eff.append([model, f"effect of +1 within-SD of PromotionSavings ({unit})", b * sd_w,
                    f"{100 * b * sd_w / ybar:.1f}% of the mean of Y ({ybar:.2f})"])
        eff.append([model, "elasticity at the means (% change in Y per 1% change in PromotionSavings)", b * xbar / ybar,
                    f"mean X = ${xbar:.2f}, mean Y = {ybar:.2f}"])
        if ycol == "disc_spend":
            z = (b - bench) / se
            eff.append([model, "proportional-allocation benchmark (discretionary share of total spend)", bench,
                        f"H0: beta = benchmark; z = {z:.2f}, p = {2 * stats.norm.sf(abs(z)):.4f}"])
    EFF = pd.DataFrame(eff, columns=["Model", "Quantity", "Value", "Note"])
    rep.say(EFF.to_string(index=False))
    rep.say("\nBenchmark note: if $1 of savings were spent in the same proportions as the existing basket, discretionary spending "
            f"would rise by about ${bench:.2f}. It is a heuristic (the control for necessity spending already absorbs part of the "
            "scale effect), not a structural test of mental accounting.")
    return EFF


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _clean_axes(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _figures(rep: Reporter, d: Data, res: dict, paths: Paths) -> None:
    rep.header("FIGURES")
    with plt.rc_context(PLOT_RC):
        # (1) coefficient on PromotionSavings across the four columns
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for ax, (title, ycol), unit in zip(axs, OUTCOMES.items(),
                                           [r"\$ per \$1 of savings", r"percentage points per \$1 of savings"]):
            xs, bs, lo, hi = np.arange(4), [], [], []
            for k in COLS:
                b, se, _ = _get(res[ycol], k, "promo_nec")
                bs.append(b)
                lo.append(b - 1.96 * se)
                hi.append(b + 1.96 * se)
            ax.axhline(0, color=INK2, linewidth=0.8)
            ax.vlines(xs, lo, hi, color=BLUE, linewidth=1.6)
            ax.plot(xs, bs, "o", color=BLUE, markersize=9, markeredgecolor="white", markeredgewidth=2)
            for x, b in zip(xs, bs):
                ax.text(x + 0.12, b, f"{b:.3f}", va="center", fontsize=8.5)
            ax.set_xticks(xs)
            ax.set_xticklabels(["(1) Pooled,\nno controls", "(a) Pooled\n+ controls", "(b) Mundlak\nCRE", "(c) Two-way\nFE"])
            ax.set_xlim(-0.5, 3.8)
            ax.set_ylabel(unit)
            ax.set_title(title.split(":")[0] + ": " + title.split(": ")[1])
            _clean_axes(ax)
        fig.suptitle("Coefficient on PromotionSavings across specifications (points) with 95% confidence intervals",
                     x=0.01, ha="left", fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(paths.out_b / "b4_fig_stability.png", dpi=250)
        plt.close(fig)

        # (2) binned partial-regression plot of the preferred (two-way FE) model
        DM = d.DM
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for ax, (title, ycol) in zip(axs, OUTCOMES.items()):
            Z = DM[["coupon", "nec_spend"]].to_numpy()
            ey = DM[ycol].to_numpy() - Z @ np.linalg.lstsq(Z, DM[ycol].to_numpy(), rcond=None)[0]
            ex = DM["promo_nec"].to_numpy() - Z @ np.linalg.lstsq(Z, DM["promo_nec"].to_numpy(), rcond=None)[0]
            bins = pd.qcut(pd.Series(ex).rank(method="first"), 40, labels=False)
            bx, by = pd.Series(ex).groupby(bins).mean(), pd.Series(ey).groupby(bins).mean()
            slope = float((ex * ey).sum() / (ex ** 2).sum())
            xx = np.array([bx.min(), bx.max()])
            ax.plot(xx, slope * xx, color=ORANGE, linewidth=1.8)
            ax.plot(bx, by, "o", color=BLUE, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
            ax.axhline(0, color=GRID, linewidth=0.8)
            ax.set_xlabel("PromotionSavings, residualised ($)")
            ax.set_ylabel(("DiscretionarySpending" if ycol == "disc_spend" else "DiscretionaryShare") + ", residualised")
            ax.set_title(title.split(":")[0] + f": slope = {slope:.3f}")
            _clean_axes(ax)
        fig.suptitle("Binned partial-regression plots (40 bins of equal size) after removing household effects, week effects, "
                     "Coupon and NecessitySpending", x=0.01, ha="left", fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(paths.out_b / "b4_fig_binned.png", dpi=250)
        plt.close(fig)
    rep.say("Saved b4_fig_stability.png and b4_fig_binned.png")
