"""
Part B, step 2: pre-estimation diagnostics (Chapter 4.4).

Reads   part_b_output/hw_main.csv and hw_all.csv                  (from step B1)
Writes  part_b_output/diagnostics_table.csv           one row per test: statistic, df, p-value, decision
        part_b_output/representativeness_table.csv    households with vs without demographics
        part_b_output/b2_distributions.png            histograms behind the transformation decision
        part_b_output/part_b2_report.txt              everything printed

Run it from main.ipynb:   b2_diagnostics.run(paths)

Specification used for the tests (the stage-(c) model of Chapter 3, without results being reported here):
    Y_it = b1*promo_nec + b2*coupon + b3*nec_spend + household effects + week effects + e_it
    Outcomes: M1  disc_spend           (levels)
              M1b asinh(disc_spend)    (transformation robustness)
              M2  disc_share           (0-100)
All fixed effects are handled by within-demeaning (alternating projections for the two-way case), which
works for the unbalanced panel and needs no dummy variables except in the Mundlak test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller

from .config import Paths
from .utils import Reporter, demean, quiet_warnings, read_panel

ALPHA = 0.05
X = ["promo_nec", "coupon", "nec_spend"]
OUTCOMES = {"M1 disc_spend": "disc_spend", "M1b asinh(disc_spend)": "asinh_disc_spend", "M2 disc_share": "disc_share"}


@dataclass
class Context:
    """Everything the sections share: the panel, its dimensions, within-transformed data and the results table."""
    rep: Reporter
    S: pd.DataFrame
    n: int = 0
    N: int = 0
    T: int = 0
    K: int = len(X)
    Ti: pd.Series | None = None
    hh_code: np.ndarray | None = None
    wk_code: np.ndarray | None = None
    DM: dict = field(default_factory=dict)
    table: list = field(default_factory=list)

    def record(self, section, test, outcome, stat, df, p, if_reject, if_not):
        """Store one test result (for diagnostics_table.csv) and print it."""
        dec = if_reject if (p is not None and p < ALPHA) else if_not
        self.table.append(dict(section=section, test=test, outcome=outcome, statistic=stat, df=df, p_value=p, decision=dec))
        ptxt = "n/a" if p is None else ("<0.0001" if p < 1e-4 else f"{p:.4f}")
        self.rep.say(f"  {outcome:<22s} stat = {stat:>12,.3f}   df = {str(df):<14s} p = {ptxt:<8s} -> {dec}")

    def ols_dm(self, dm, y):
        """OLS of y on X using already within-transformed data; returns coefficients, residuals, RSS."""
        Y, Z = dm[y].to_numpy(), dm[X].to_numpy()
        b = np.linalg.lstsq(Z, Y, rcond=None)[0]
        e = Y - Z @ b
        return b, e, float(e @ e)


@quiet_warnings()
def run(paths: Paths) -> pd.DataFrame:
    paths.make_dirs()
    rep = Reporter()
    S = read_panel(paths.hw_main)
    S["asinh_disc_spend"] = np.arcsinh(S["disc_spend"])

    ctx = Context(rep=rep, S=S, n=len(S), N=S["household_key"].nunique(), T=S["week_no"].nunique())
    ctx.hh_code = pd.factorize(S["household_key"])[0]
    ctx.wk_code = pd.factorize(S["week_no"])[0]

    _panel_structure(ctx)
    _distributions(ctx, paths)
    allc = list(dict.fromkeys(list(OUTCOMES.values()) + X))
    ctx.DM = {"tw": demean(S, allc, ["household_key", "week_no"]),
              "hh": demean(S, allc, ["household_key"]),
              "wk": demean(S, allc, ["week_no"])}
    _multicollinearity(ctx)
    _unit_roots(ctx)
    _poolability(ctx)
    _residual_diagnostics(ctx, allc)
    _inference_choice(ctx)
    _representativeness(ctx, paths)
    return _summary_table(ctx, paths)


def _vif(M: np.ndarray) -> list[float]:
    M = M - M.mean(axis=0)
    out = []
    for j in range(M.shape[1]):
        y, Z = M[:, j], np.delete(M, j, axis=1)
        b = np.linalg.lstsq(Z, y, rcond=None)[0]
        out.append(1 / (1 - (1 - ((y - Z @ b) ** 2).sum() / (y ** 2).sum())))
    return out


def _pesaran_cd(resid, hh_i, wk_i, min_overlap=10):
    """Pesaran (2015) CD statistic for an unbalanced panel (pairs with at least ``min_overlap`` common weeks)."""
    Nn, Tt = hh_i.max() + 1, wk_i.max() + 1
    R, M = np.zeros((Nn, Tt)), np.zeros((Nn, Tt))
    R[hh_i, wk_i], M[hh_i, wk_i] = resid, 1
    A, B, Tij = R @ R.T, (R ** 2) @ M.T, M @ M.T
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = A / np.sqrt(B * B.T)
    iu = np.triu_indices(Nn, 1)
    ok = (Tij[iu] >= min_overlap) & np.isfinite(rho[iu])
    r, t = rho[iu][ok], Tij[iu][ok]
    cd = float((np.sqrt(t) * r).sum() / np.sqrt(len(r)))
    return cd, float(r.mean()), float(np.abs(r).mean()), int(len(r))


# ---------------------------------------------------------------------------
def _panel_structure(c: Context) -> None:
    S, say = c.S, c.rep.say
    c.rep.header("1. PANEL STRUCTURE")
    c.Ti = S.groupby("household_key").size()
    Ti = c.Ti
    say(f"Observations (household-weeks): {c.n:,}   households N = {c.N}   distinct weeks T = {c.T}   regressors K = {c.K}")
    say(f"Observations per household: min {Ti.min()}, p25 {Ti.quantile(.25):.0f}, median {Ti.median():.0f}, "
        f"p75 {Ti.quantile(.75):.0f}, max {Ti.max()}")
    say(f"Balance: {c.n:,} of {c.N * c.T:,} possible household-weeks observed ({c.n / (c.N * c.T):.1%}) -> unbalanced panel")
    g = S.groupby("household_key")["week_no"].apply(lambda w: (w.diff().dropna() > 1).sum())
    say(f"Households with at least one gap between observed weeks: {(g > 0).mean():.1%}; "
        f"median number of gaps per household: {g.median():.0f}")
    say("Missing values in model variables:", int(S[X + ['disc_spend', 'disc_share']].isna().sum().sum()))
    say("Decision: unbalanced panel is acceptable for fixed-effects estimation (within estimator needs no balance); "
        "tests below use unbalanced-panel formulas.")


def _distributions(c: Context, paths: Paths) -> None:
    S, say = c.S, c.rep.say
    c.rep.header("2. DISTRIBUTIONS AND TRANSFORMATIONS")
    S["asinh_promo_nec"] = np.arcsinh(S["promo_nec"])
    rows = []
    for col in ["disc_spend", "asinh_disc_spend", "disc_share", "total_spend", "nec_spend", "promo_nec",
                "asinh_promo_nec", "promo_all", "coupon"]:
        x = S[col]
        rows.append((col, round(x.mean(), 2), round(x.std(), 2), round(stats.skew(x), 2), round(stats.kurtosis(x), 2),
                     round((x <= 0).mean(), 3), round(x.quantile(.01), 2), round(x.quantile(.99), 2)))
    dist = pd.DataFrame(rows, columns=["variable", "mean", "sd", "skewness", "excess_kurtosis", "share_zero", "p1", "p99"])
    say(dist.to_string(index=False))
    say(f"\nSkewness of disc_spend is {stats.skew(S['disc_spend']):.2f} in levels and {stats.skew(S['asinh_disc_spend']):.2f} "
        "after the inverse hyperbolic sine transform (compare absolute values, and see the histograms).")
    say("Decision: if the spending outcomes are right-skewed with a mass at zero (check skewness and share_zero above) -> keep "
        "levels (dollars, easy to interpret) as the baseline and report asinh() as a robustness check; log(x) is not "
        "usable because of the zeros.")

    say("\nShare outcome (0-100) and small-basket noise:")
    say(f"  disc_share == 0   : {(S.disc_share <= 0).mean():.1%}     disc_share == 100: {(S.disc_share >= 100).mean():.1%}")
    for thr in [5, 10, 20]:
        say(f"  household-weeks with total_spend < ${thr:>2}: {(S.total_spend < thr).mean():.1%}   "
            f"(of which disc_share is 0 or 100: {((S.total_spend < thr) & ((S.disc_share <= 0) | (S.disc_share >= 100))).sum() / max((S.total_spend < thr).sum(), 1):.1%})")
    say("Decision: shares from very small baskets are mechanically 0 or 100 -> estimate Model 2 on all weeks, and as a "
        "robustness check restrict to total_spend >= $10 and/or weight by total_spend; add a fractional-response model.")

    fig, ax = plt.subplots(2, 3, figsize=(12, 6.4))
    panels = [("disc_spend", "DiscretionarySpending ($)"), ("asinh_disc_spend", "asinh(DiscretionarySpending)"),
              ("disc_share", "DiscretionaryShare (%)"), ("promo_nec", "PromotionSavings, necessity ($)"),
              ("coupon", "Coupon value ($), log y-axis"), ("nec_spend", "NecessitySpending ($)")]
    for a, (col, lab) in zip(ax.ravel(), panels):
        a.hist(S[col], bins=60, color="#4C78A8", edgecolor="white", linewidth=0.3)
        a.set_title(lab, fontsize=10)
        a.tick_params(labelsize=8)
        if col == "coupon":
            a.set_yscale("log")
    fig.suptitle("Distributions of the main variables (household-weeks, main sample)", fontsize=11)
    fig.tight_layout()
    fig.savefig(paths.out_b / "b2_distributions.png", dpi=200)
    plt.close(fig)
    say("Saved figure b2_distributions.png")


def _multicollinearity(c: Context) -> None:
    S, say, DM = c.S, c.rep.say, c.DM
    c.rep.header("3. MULTICOLLINEARITY (VIF of the three regressors)")
    v_pool, v_within = _vif(S[X].to_numpy()), _vif(DM["tw"][X].to_numpy())
    say(pd.DataFrame({"regressor": X, "VIF_pooled": np.round(v_pool, 2), "VIF_within(two-way)": np.round(v_within, 2)})
        .to_string(index=False))
    say("Correlation of the regressors after removing household and week effects:")
    say(DM["tw"][X].corr().round(2).to_string())
    mx = max(max(v_pool), max(v_within))
    c.table.append(dict(section="Multicollinearity", test="Max VIF (rule of thumb 5-10)", outcome="regressors", statistic=mx,
                        df="", p_value=None,
                        decision="Acceptable (< 5)" if mx < 5 else "Concern (>= 5): report models with/without NecessitySpending"))
    say(f"Decision: max VIF = {mx:.2f} -> " + ("acceptable (< 5)." if mx < 5 else "report models with and without NecessitySpending."))


def _unit_roots(c: Context) -> None:
    S, say = c.S, c.rep.say
    c.rep.header("4. PANEL UNIT-ROOT CHECK (Fisher-type ADF, indicative)")
    say("Households with >= 40 observed weeks; series ordered by week (gaps ignored); ADF with constant, 2 lags.")
    for lab, col in [("disc_spend", "disc_spend"), ("disc_share", "disc_share"), ("promo_nec", "promo_nec")]:
        pv = []
        for _, gdf in S.groupby("household_key"):
            x = gdf[col].to_numpy()
            if len(x) >= 40 and np.std(x) > 0:
                try:
                    pv.append(adfuller(x, maxlag=2, autolag=None, regression="c")[1])
                except Exception:
                    pass
        pv = np.clip(np.array(pv), 1e-300, 1)
        stat = -2 * np.log(pv).sum()
        p = float(stats.chi2.sf(stat, 2 * len(pv)))
        c.record("Unit root", f"Fisher ADF (H0: all series have a unit root), k={len(pv)} hh", lab, stat, 2 * len(pv), p,
                 "Reject: series are stationary -> levels are fine", "Cannot reject: consider first differences")
        say(f"      share of households rejecting a unit root individually at 5%: {(pv < .05).mean():.1%}")
    say("Note: the panel is unbalanced with gaps, so this is indicative; time fixed effects absorb common trends.")


def _poolability(c: Context) -> None:
    S, say, DM = c.S, c.rep.say, c.DM
    n, N, T, K = c.n, c.N, c.T, c.K
    c.rep.header("5. POOLABILITY AND MODEL CHOICE")
    say("F-test for household effects (pooled OLS + week effects  vs  two-way FE):")
    for lab, y in OUTCOMES.items():
        rss_a, rss_b = c.ols_dm(DM["wk"], y)[2], c.ols_dm(DM["tw"], y)[2]
        d1, d2 = N - 1, n - N - (T - 1) - K
        F = ((rss_a - rss_b) / d1) / (rss_b / d2)
        c.record("Model choice", "F-test: household effects", lab, F, f"({d1}, {d2})", float(stats.f.sf(F, d1, d2)),
                 "Reject: household effects needed (FE over pooled)", "Do not reject: pooled OLS adequate")
    say("\nF-test for time effects (household FE  vs  two-way FE):")
    for lab, y in OUTCOMES.items():
        rss_a, rss_b = c.ols_dm(DM["hh"], y)[2], c.ols_dm(DM["tw"], y)[2]
        d1, d2 = T - 1, n - N - (T - 1) - K
        F = ((rss_a - rss_b) / d1) / (rss_b / d2)
        c.record("Model choice", "F-test: week effects", lab, F, f"({d1}, {d2})", float(stats.f.sf(F, d1, d2)),
                 "Reject: keep week fixed effects", "Do not reject: week effects not needed")

    say("\nBreusch-Pagan LM test for random household effects (Baltagi-Li form for unbalanced panels; pooled OLS + week effects):")
    Tsum = float((c.Ti * (c.Ti - 1)).sum())
    for lab, y in OUTCOMES.items():
        e = c.ols_dm(DM["wk"], y)[1]
        s1 = float(pd.Series(e).groupby(c.hh_code).sum().pow(2).sum())
        s2 = float((e ** 2).sum())
        lm = (n ** 2) / (2 * Tsum) * (s1 / s2 - 1) ** 2
        c.record("Model choice", "Breusch-Pagan LM: variance of household effect = 0", lab, lm, "1", float(stats.chi2.sf(lm, 1)),
                 "Reject: panel model needed over pooled OLS", "Do not reject: pooled OLS adequate")

    say("\nMundlak (correlated random effects) test = robust Hausman: joint test that the household means of the "
        "regressors are zero, SEs clustered by household:")
    Zb = S.groupby("household_key")[X].transform("mean").add_suffix("_bar")
    W = pd.get_dummies(S["week_no"], prefix="w", drop_first=True, dtype=float)
    for lab, y in OUTCOMES.items():
        D = sm.add_constant(pd.concat([S[X], Zb, W], axis=1))
        r = sm.OLS(S[y], D).fit(cov_type="cluster", cov_kwds={"groups": c.hh_code})
        R = np.zeros((K, D.shape[1]))
        for k, col in enumerate(Zb.columns):
            R[k, list(D.columns).index(col)] = 1
        w = r.wald_test(R, use_f=False, scalar=True)
        c.record("Model choice", "Mundlak test: household means = 0", lab, float(np.squeeze(w.statistic)), str(K),
                 float(np.squeeze(w.pvalue)),
                 "Reject: random effects inconsistent -> use fixed effects", "Do not reject: random effects acceptable")
    say("(The classical Hausman test is invalid with clustered errors, which is why the Mundlak version is used.)")


def _residual_diagnostics(c: Context, allc: list[str]) -> None:
    S, say, DM = c.S, c.rep.say, c.DM
    n, K = c.n, c.K
    c.rep.header("6. SERIAL CORRELATION, HETEROSKEDASTICITY, CROSS-SECTIONAL DEPENDENCE")
    say("Wooldridge (2002) first-difference test for serial correlation (H0: no serial correlation, rho = -0.5):")
    D1 = S[["household_key", "week_no"] + allc].copy()
    lagk = D1.groupby("household_key").shift(1)
    cons = (D1["week_no"] - lagk["week_no"]) == 1
    for lab, y in OUTCOMES.items():
        d = pd.DataFrame({"household_key": D1["household_key"], "week_no": D1["week_no"]})
        for col in [y] + X:
            d["d_" + col] = D1[col] - lagk[col]
        d = d[cons].dropna().copy()
        Zd, Yd = d[["d_" + col for col in X]].to_numpy(), d["d_" + y].to_numpy()
        b = np.linalg.lstsq(Zd, Yd, rcond=None)[0]
        d["e"] = Yd - Zd @ b
        lag = d[["household_key", "week_no", "e"]].copy()
        lag["week_no"] += 1
        dd = d.merge(lag, on=["household_key", "week_no"], suffixes=("", "_lag")).dropna()
        r = sm.OLS(dd["e"].to_numpy(), dd[["e_lag"]].to_numpy()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["household_key"])[0]})
        rho, se = float(r.params[0]), float(r.bse[0])
        tt = (rho + 0.5) / se
        G = dd["household_key"].nunique()
        p = float(2 * stats.t.sf(abs(tt), G - 1))
        c.record("Residual diagnostics", f"Wooldridge serial correlation (rho = {rho:.3f})", lab, tt, f"t, {G - 1}", p,
                 "Reject: serial correlation present -> cluster SEs by household", "Do not reject: no serial correlation")

    say("\nBreusch-Pagan / Koenker heteroskedasticity test on two-way FE residuals:")
    for lab, y in OUTCOMES.items():
        b, e, _ = c.ols_dm(DM["tw"], y)
        lm, p_lm, _, _ = het_breuschpagan(e, np.column_stack([np.ones(n), DM["tw"][X].to_numpy()]))
        c.record("Residual diagnostics", "Breusch-Pagan (Koenker) heteroskedasticity", lab, float(lm), str(K), float(p_lm),
                 "Reject: heteroskedasticity -> robust (clustered) SEs", "Do not reject: homoskedastic")

    say("\nPesaran (2015) CD test for cross-sectional dependence, on household-FE residuals (pairs with >= 10 common weeks).")
    say("This asks whether households share weekly shocks (holidays, promotion calendars) that week effects should absorb.")
    for lab, y in OUTCOMES.items():
        e = c.ols_dm(DM["hh"], y)[1]
        cd, mr, ma, npair = _pesaran_cd(e, c.hh_code, c.wk_code)
        p = float(2 * stats.norm.sf(abs(cd)))
        c.record("Residual diagnostics", f"Pesaran CD, household-FE residuals (mean rho {mr:.3f})", lab, cd, f"{npair:,} pairs", p,
                 "Reject: common weekly shocks -> keep week FE and report two-way clustered SEs",
                 "Do not reject: little common-shock dependence")
    say("(Not repeated after two-way FE: removing week effects forces residuals to sum to ~zero within each week, which "
        "mechanically makes the CD statistic negative. Section 7 checks inference with week and two-way clustering instead.)")


def _inference_choice(c: Context) -> None:
    say, DM = c.rep.say, c.DM
    n, N, T, K = c.n, c.N, c.T, c.K
    c.rep.header("7. INFERENCE CHOICE: standard errors under alternative assumptions (two-way FE, M1)")
    y = "disc_spend"
    dm = DM["tw"]
    Zd = sm.add_constant(dm[X], has_constant="add").drop(columns="const")
    base = sm.OLS(dm[y].to_numpy(), Zd.to_numpy())
    dof_fix = np.sqrt((n - K) / (n - N - (T - 1) - K))
    se = {}
    se["classical (dof-corrected)"] = base.fit().bse * dof_fix
    se["heteroskedasticity-robust (HC1)"] = base.fit(cov_type="HC1").bse
    se["cluster: household"] = base.fit(cov_type="cluster", cov_kwds={"groups": c.hh_code}).bse
    se["cluster: week"] = base.fit(cov_type="cluster", cov_kwds={"groups": c.wk_code}).bse
    try:
        se["two-way cluster: household and week"] = base.fit(
            cov_type="cluster", cov_kwds={"groups": np.column_stack([c.hh_code, c.wk_code])}).bse
    except Exception as ex:
        say("  (two-way clustering not available:", ex, ")")
    tab = pd.DataFrame(se, index=X).T
    tab_ratio = (tab / tab.loc["cluster: household"]).round(2)
    say("Standard errors:")
    say(tab.round(4).to_string())
    say("\nRatio to the household-clustered SE (values > 1 mean clustering by household is less conservative):")
    say(tab_ratio.to_string())
    say("Decision: household-clustered SEs are the baseline (Bertrand, Duflo & Mullainathan, 2004); report two-way "
        "(household and week) clustered SEs as a check.")


def _representativeness(c: Context, paths: Paths) -> None:
    say = c.rep.say
    c.rep.header("8. REPRESENTATIVENESS: households with vs without demographics (clean panel, all households >= 10 active weeks)")
    A = pd.read_csv(paths.hw_all)
    hhA = (A.groupby("household_key")
             .agg(has_demo=("has_demo", "max"), active_weeks=("week_no", "nunique"), mean_weekly_spend=("total_spend", "mean"),
                  mean_disc_share=("disc_share", "mean"), mean_promo_nec=("promo_nec", "mean"),
                  share_weeks_coupon=("coupon_any", "mean"), baskets=("n_baskets", "sum")).reset_index())
    rows = []
    for col in ["active_weeks", "baskets", "mean_weekly_spend", "mean_disc_share", "mean_promo_nec", "share_weeks_coupon"]:
        a, b = hhA.loc[hhA.has_demo == 1, col], hhA.loc[hhA.has_demo == 0, col]
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append((col, round(a.mean(), 2), round(b.mean(), 2), round((a.mean() - b.mean()) / sd, 2)))
    rep_table = pd.DataFrame(rows, columns=["variable", "with_demographics", "without", "standardised_diff"])
    say(f"Households: with demographics = {int((hhA.has_demo == 1).sum())}, without = {int((hhA.has_demo == 0).sum())}")
    say(rep_table.to_string(index=False))
    rep_table.to_csv(paths.out_b / "representativeness_table.csv", index=False)


def _summary_table(c: Context, paths: Paths) -> pd.DataFrame:
    say = c.rep.say
    c.rep.header("9. SUMMARY TABLE FOR CHAPTER 4.4")
    tb = pd.DataFrame(c.table)
    tb.to_csv(paths.out_b / "diagnostics_table.csv", index=False)
    show = tb.copy()
    show["p_value"] = show["p_value"].apply(lambda p: "" if p is None or pd.isna(p) else ("<0.0001" if p < 1e-4 else f"{p:.4f}"))
    show["statistic"] = show["statistic"].apply(lambda v: f"{v:,.2f}")
    say(show[["test", "outcome", "statistic", "df", "p_value", "decision"]].to_string(index=False))
    say("\nSaved diagnostics_table.csv, representativeness_table.csv, b2_distributions.png in", paths.show(paths.out_b))
    c.rep.save(paths.out_b / "part_b2_report.txt")
    return tb
