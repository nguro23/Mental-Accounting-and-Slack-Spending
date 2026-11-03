"""
Part B, step 5: robustness, falsification and mechanism checks (Chapter 4.8).

Reads   part_b_output/hw_main.csv, hw_all.csv, trips_main.csv      (from step B1)
Writes  part_b_output/b5_tables.xlsx        all results (long format) plus one sheet per block
        part_b_output/b5_summary.csv        sign counts across specifications
        part_b_output/part_b5_report.txt    everything printed

Run it from main.ipynb:   b5_robustness.run(paths)

Every estimate is a household-clustered regression. "Two-way FE" = household + week fixed effects.
Blocks
  A  Build-up: which ingredient turns the raw positive association into the preferred estimate?
  B  Alternative basket-size control (shelf vs net), alternative savings measures (all items, markdown depth)
  C  Coupon definitions (binary, winsorised)
  D  Outcome definitions (narrow, broad, without sugary drinks, without alcohol, asinh)
  E  Samples (baskets >= $10, <= 10 trips, winsorised, >= 52 weeks, all households, household-trip panel)
  F  Placebo (lead savings), dynamics (lag), falsification outcome (Neutral goods)
  G  Where does the money go? (savings and coupons on each spending category, no necessity control)
  H  Heterogeneity (income, household size, children, first vs second half of the panel)
  I  Non-linearity (quartiles of savings)
  J  Fractional-response (logit) model for the share, Mundlak form
  K  Coupons split by item category (needs coupon_nec / coupon_disc from step B1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import Paths
from .utils import Reporter, demean, quiet_warnings, read_panel, star

XV = ["promo_nec", "coupon", "nec_spend"]
DEMO = ["income_10k", "hh_size_num", "has_kids"]
TW = ("household_key", "week_no")
Y1, Y2 = "disc_spend", "disc_share"


class Estimator:
    """Runs household-clustered regressions, stores every coefficient in ``rows`` and prints them block by block."""

    def __init__(self, rep: Reporter) -> None:
        self.rep = rep
        self.rows: list[dict] = []
        # variables shown in the printed tables (the others, e.g. controls, are stored but not printed)
        self.focal = {"promo_nec", "promo_all", "promo_nec_rate", "coupon", "coupon_any", "coupon_w", "promo_nec_lead",
                      "promo_nec_lag", "sav_q2", "sav_q3", "sav_q4"}

    def run(self, block, spec, df, y, xs, absorb=TW, outcome=None, summary=False) -> None:
        d = df.dropna(subset=[y] + xs).reset_index(drop=True)
        if len(d) < 200 or d["household_key"].nunique() < 20:
            return
        dm = demean(d, [y] + xs, list(absorb))
        g = pd.factorize(d["household_key"])[0]
        r = sm.OLS(dm[y].to_numpy(), dm[xs].to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": g})
        for i, v in enumerate(xs):
            self.rows.append(dict(Block=block, Specification=spec, Outcome=outcome or y, Variable=v, Coef=float(r.params[i]),
                                  SE=float(r.bse[i]), p=float(r.pvalues[i]), N=len(d),
                                  Households=d["household_key"].nunique(), Summary=summary))

    def show(self, block, focal=True) -> None:
        t = pd.DataFrame([r for r in self.rows if r["Block"] == block])
        if t.empty:
            self.rep.say("(no estimates)")
            return
        if focal:
            t = t[t["Variable"].isin(self.focal)]
        t = t.assign(Estimate=t.apply(lambda r: f"{r.Coef:.3f}{star(r.p)}", axis=1),
                     SE_=t["SE"].map(lambda v: f"({v:.3f})"), p_=t["p"].map(lambda v: "<0.001" if v < 1e-3 else f"{v:.3f}"),
                     N_=t["N"].map(lambda v: f"{v:,}"))
        self.rep.say(t[["Specification", "Outcome", "Variable", "Estimate", "SE_", "p_", "N_"]]
                     .rename(columns={"SE_": "SE", "p_": "p", "N_": "N"}).to_string(index=False))


@quiet_warnings()
def run(paths: Paths) -> pd.DataFrame:
    paths.make_dirs()
    rep = Reporter()
    S = read_panel(paths.hw_main)
    A = read_panel(paths.hw_all)
    TR = read_panel(paths.trips_main)

    S["income_10k"] = S["income_k_mid"] / 10.0
    S["other_disc_spend"] = S["disc_broad_spend"] - S["disc_spend"]
    S["asinh_disc"] = np.arcsinh(S["disc_spend"])
    S["disc_xsugar_share"] = 100 * S["disc_xsugar_spend"] / S["total_spend"]
    S["disc_xalc_share"] = 100 * S["disc_xalc_spend"] / S["total_spend"]
    S["coupon_w"] = S["coupon"].clip(upper=S["coupon"].quantile(0.99))
    S["income_grp"] = pd.cut(S["income_k_mid"], [0, 35, 75, 1e6], right=False, labels=["< $35K", "$35-74K", "$75K+"])
    S["size_grp"] = np.where(S["hh_size_num"] >= 3, "3+ persons", np.where(S["hh_size_num"] == 2, "2 persons", "1 person"))
    S["kids_grp"] = np.where(S["has_kids"] == 1, "Children (1+)", "None / unknown")

    est = Estimator(rep)
    _block_a(est, S)
    _block_b(est, S)
    _block_c(est, S)
    _block_d(est, S)
    _block_e(est, S, A, TR)
    _block_f(est, S)
    _block_g(est, S)
    _block_h(est, S)
    _block_i(est, S)
    _block_j(est, S)
    _block_k(est, S)
    return _summary_and_save(est, paths)


# ---------------------------------------------------------------------------
def _block_a(est: Estimator, S) -> None:
    say = est.rep.say
    est.rep.header("A. BUILD-UP: from the raw association to the preferred specification (main sample)")
    say("Which ingredient removes the positive raw association? Same two outcomes, savings coefficient shown.")
    steps = [("A1 no controls, no FE", [], ["promo_nec", "coupon"]),
             ("A2 + week FE", ["week_no"], ["promo_nec", "coupon"]),
             ("A3 + household FE", ["household_key"], ["promo_nec", "coupon"]),
             ("A4 two-way FE, no necessity control", list(TW), ["promo_nec", "coupon"]),
             ("A5 no FE, + NecessitySpending", [], XV),
             ("A6 two-way FE + NecessitySpending (preferred)", list(TW), XV)]
    for y in (Y1, Y2):
        for name, ab, xs in steps:
            est.run("A", name, S, y, xs, absorb=tuple(ab))
    est.show("A")


def _block_b(est: Estimator, S) -> None:
    say = est.rep.say
    est.rep.header("B. BASKET-SIZE CONTROL AND SAVINGS MEASURE (two-way FE, main sample)")
    say("Net vs shelf control: NecessitySpending is measured NET of markdowns. Holding it fixed while savings rise means the")
    say("household bought MORE necessities at shelf prices. The shelf-price control (net + savings) holds the planned basket fixed.")
    say("Algebra: beta(savings | shelf control) = beta(savings | net control) - beta(NecessitySpending | net control).")
    for y in (Y1, Y2):
        est.run("B", "B1 preferred: net NecessitySpending", S, y, XV, summary=True)
        est.run("B", "B2 shelf-price NecessitySpending control", S, y, ["promo_nec", "coupon", "nec_spend_shelf"], summary=True)
        est.run("B", "B3 no necessity control", S, y, ["promo_nec", "coupon"], summary=True)
        est.run("B", "B4 savings on ALL items (secondary measure)", S, y, ["promo_all", "coupon", "nec_spend"], summary=True)
        est.run("B", "B5 markdown depth %, shelf control", S, y, ["promo_nec_rate", "coupon", "nec_spend_shelf"], summary=True)
    est.show("B")
    say("Note: B4 is mechanically linked to discretionary spending (it includes savings ON discretionary items); B5 uses the")
    say("markdown DEPTH (% of shelf value), which varies independently of basket size.")


def _block_c(est: Estimator, S) -> None:
    est.rep.header("C. COUPON DEFINITIONS (two-way FE)")
    for y in (Y1, Y2):
        est.run("C", "C1 coupon value ($), preferred", S, y, XV, summary=True)
        est.run("C", "C2 coupon redeemed (0/1)", S, y, ["promo_nec", "coupon_any", "nec_spend"], summary=True)
        est.run("C", "C3 coupon value winsorised at p99", S, y, ["promo_nec", "coupon_w", "nec_spend"], summary=True)
    est.show("C")


def _block_d(est: Estimator, S) -> None:
    est.rep.header("D. OUTCOME DEFINITIONS (two-way FE, preferred controls)")
    outs = [("disc_narrow_spend", "narrow definition ($)"), ("disc_broad_spend", "broad definition ($)"),
            ("disc_xsugar_spend", "excluding sugary drinks ($)"), ("disc_xalc_spend", "excluding alcohol ($)"),
            ("asinh_disc", "asinh(DiscretionarySpending)"),
            ("disc_narrow_share", "narrow definition (pp)"), ("disc_broad_share", "broad definition (pp)"),
            ("disc_xsugar_share", "excluding sugary drinks (pp)"), ("disc_xalc_share", "excluding alcohol (pp)")]
    for y, lab in outs:
        est.run("D", f"D {lab}", S, y, XV, outcome=y, summary=True)
    est.show("D")


def _block_e(est: Estimator, S, A, TR) -> None:
    est.rep.header("E. SAMPLES (two-way FE, preferred controls)")
    samples = {
        "E1 weeks with total spend >= $10": S[S["total_spend"] >= 10],
        "E2 weeks with <= 10 trips": S[S["n_baskets"] <= 10],
        "E4 households with >= 52 active weeks": S[S.groupby("household_key")["week_no"].transform("nunique") >= 52],
    }
    W = S.copy()
    for c in [Y1, Y2, "promo_nec", "coupon", "nec_spend"]:
        W[c] = W[c].clip(W[c].quantile(.01), W[c].quantile(.99))
    samples["E3 variables winsorised at 1% / 99%"] = W
    for name, df in samples.items():
        for y in (Y1, Y2):
            est.run("E", name, df, y, XV, summary=True)
    for y in (Y1, Y2):
        est.run("E", "E5 ALL households >= 10 weeks (no demographics)", A, y, XV, summary=True)
        est.run("E", "E6 household-trip (basket) panel", TR, y, XV, summary=True)
    est.show("E")


def _block_f(est: Estimator, S) -> None:
    est.rep.header("F. PLACEBO, DYNAMICS AND FALSIFICATION (two-way FE)")
    lead = S[["household_key", "week_no", "promo_nec"]].copy()
    lead["week_no"] -= 1
    lead = lead.rename(columns={"promo_nec": "promo_nec_lead"})
    lag = S[["household_key", "week_no", "promo_nec"]].copy()
    lag["week_no"] += 1
    lag = lag.rename(columns={"promo_nec": "promo_nec_lag"})
    SL = S.merge(lead, on=["household_key", "week_no"], how="left").merge(lag, on=["household_key", "week_no"], how="left")
    for y in (Y1, Y2):
        est.run("F", "F1 placebo: add NEXT week's savings (should be ~0)", SL, y, ["promo_nec", "promo_nec_lead", "coupon", "nec_spend"])
        est.run("F", "F2 add previous and next week's savings", SL, y, ["promo_nec", "promo_nec_lag", "promo_nec_lead", "coupon", "nec_spend"])
    est.run("F", "F3 falsification: Neutral spending ($)", S, "neutral_spend", XV)
    est.run("F", "F4 falsification: Neutral share (pp)", S, "neutral_share", XV)
    est.show("F")


def _block_g(est: Estimator, S) -> None:
    say = est.rep.say
    est.rep.header("G. WHERE DOES THE MONEY GO? effect of $1 savings / coupon on each category (two-way FE, NO necessity control)")
    cats = [("total_spend", "Total merchandise spend"), ("nec_spend", "Necessity spend"), ("disc_spend", "Discretionary (main)"),
            ("other_disc_spend", "Discretionary (extended)"), ("neutral_spend", "Neutral spend")]
    for y, lab in cats:
        est.run("G", f"G {lab}", S, y, ["promo_nec", "coupon"], outcome=y)
    est.show("G")
    g = pd.DataFrame([r for r in est.rows if r["Block"] == "G" and r["Variable"] == "promo_nec"]).set_index("Outcome")["Coef"]
    say(f"\nCheck (identity): necessity + main disc + extended disc + neutral = {g[['nec_spend','disc_spend','other_disc_spend','neutral_spend']].sum():.4f}"
        f"  vs total_spend = {g['total_spend']:.4f}")
    say("Net necessity spend can FALL with savings (net of markdowns), so read 'total' and 'necessity' together with the")
    say("shelf-price view: shelf necessity spend = net + savings.")


def _block_h(est: Estimator, S) -> None:
    est.rep.header("H. HETEROGENEITY (two-way FE within each subsample, preferred controls)")
    for col in ["income_grp", "size_grp", "kids_grp"]:
        for k, d in S.groupby(col, observed=True):
            for y in (Y1, Y2):
                est.run("H", f"H {k}", d, y, XV, summary=True)
    for lab, d in [("H weeks 1-51", S[S["week_no"] <= 51]), ("H weeks 52-102", S[S["week_no"] >= 52])]:
        for y in (Y1, Y2):
            est.run("H", lab, d, y, XV, summary=True)
    est.show("H")


def _block_i(est: Estimator, S) -> None:
    est.rep.header("I. NON-LINEARITY: quartiles of PromotionSavings (lowest quartile = reference; two-way FE)")
    S["sav_q"] = pd.qcut(S["promo_nec"], 4, labels=False, duplicates="drop")
    for q in (1, 2, 3):
        S[f"sav_q{q + 1}"] = (S["sav_q"] == q).astype(float)
    est.rep.say("Quartile cut-points of promo_nec ($):", S["promo_nec"].quantile([.25, .5, .75]).round(2).tolist())
    for y in (Y1, Y2):
        est.run("I", "I quartile dummies", S, y, ["sav_q2", "sav_q3", "sav_q4", "coupon", "nec_spend"])
    est.show("I")


def _block_j(est: Estimator, S) -> None:
    say = est.rep.say
    est.rep.header("J. FRACTIONAL-RESPONSE MODEL FOR THE SHARE (logit, Mundlak form, household-clustered)")
    hh = pd.factorize(S["household_key"])[0]
    Wd = pd.get_dummies(S["week_no"], prefix="w", drop_first=True, dtype=float)
    jrows = []
    for lab, ctrl in [("net NecessitySpending", "nec_spend"), ("shelf NecessitySpending", "nec_spend_shelf")]:
        xs = ["promo_nec", "coupon", ctrl]
        Xb = S.groupby("household_key")[xs].transform("mean").add_suffix("_bar")
        Z = sm.add_constant(pd.concat([S[xs + DEMO], Xb, Wd], axis=1))
        try:
            glm = sm.GLM(S["disc_share"] / 100, Z, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": hh}, maxiter=200)
            mu = glm.predict(Z)
            f = 100 * float(np.mean(mu * (1 - mu)))
            for v in ("promo_nec", "coupon"):
                jrows.append([lab, v, glm.params[v], glm.bse[v], glm.pvalues[v], f * glm.params[v], f * glm.bse[v]])
        except Exception as ex:
            say(f"  fractional logit ({lab}) failed: {ex}")
    J = pd.DataFrame(jrows, columns=["Control", "Variable", "Logit coef", "SE", "p", "APE (pp per $1)", "APE SE (approx.)"])
    say(J.round(4).to_string(index=False))
    say("APE = average partial effect on the share in percentage points; its SE here is approximate (ignores estimation of the "
        "scale factor). Compare with the linear estimates in blocks A and B.")
    for r in jrows:
        est.rows.append(dict(Block="J", Specification=f"J fractional logit APE, {r[0]}", Outcome="disc_share (pp)", Variable=r[1],
                             Coef=r[5], SE=r[6], p=r[4], N=len(S), Households=S["household_key"].nunique(), Summary=False))


def _block_k(est: Estimator, S) -> None:
    say = est.rep.say
    est.rep.header("K. COUPONS BY ITEM CATEGORY (two-way FE)")
    say("Coupons attach to specific items, so a coupon on a discretionary item is mechanically part of discretionary spending.")
    say("If the positive coupon coefficient comes from coupons on discretionary items, it is not evidence of reallocation.")
    if {"coupon_nec", "coupon_disc"} <= set(S.columns):
        S["coupon_other"] = S["coupon"] - S["coupon_nec"] - S["coupon_disc"]
        for y in (Y1, Y2):
            est.run("K", "K1 coupons on necessity items only", S, y, ["promo_nec", "coupon_nec", "nec_spend"], outcome=y)
            est.run("K", "K2 coupons on necessity and discretionary items", S, y, ["promo_nec", "coupon_nec", "coupon_disc", "nec_spend"], outcome=y)
        est.focal.update({"coupon_nec", "coupon_disc"})
        est.show("K")
    else:
        say("coupon_nec / coupon_disc not in the panel: re-run step B1 (b1_build_panel) first, then this step. Block K skipped.")


def _summary_and_save(est: Estimator, paths: Paths) -> pd.DataFrame:
    say = est.rep.say
    est.rep.header("SUMMARY: how many robustness specifications show a positive, negative or null effect?")
    R = pd.DataFrame(est.rows)
    sm_rows = R[R["Summary"]]
    out = []
    for lab, vars_ in [("Savings (promo_nec / promo_all / markdown depth)", {"promo_nec", "promo_all", "promo_nec_rate"}),
                       ("Coupon (value / binary / winsorised)", {"coupon", "coupon_any", "coupon_w"})]:
        t = sm_rows[sm_rows["Variable"].isin(vars_)]
        is_share = t["Outcome"].str.contains("share")
        for model, tt in [("levels / dollars", t[~is_share]), ("shares / pp", t[is_share])]:
            out.append([lab, model, len(tt), int(((tt.Coef > 0) & (tt.p < .05)).sum()),
                        int(((tt.Coef < 0) & (tt.p < .05)).sum()), int((tt.p >= .05).sum())])
    SUMM = pd.DataFrame(out, columns=["Variable group", "Outcome type", "Specifications", "Positive & p<.05",
                                      "Negative & p<.05", "Not significant"])
    say(SUMM.to_string(index=False))
    SUMM.to_csv(paths.out_b / "b5_summary.csv", index=False)

    with pd.ExcelWriter(paths.out_b / "b5_tables.xlsx") as xw:
        R.to_excel(xw, sheet_name="All", index=False)
        for b in sorted(R["Block"].unique()):
            R[R["Block"] == b].to_excel(xw, sheet_name=f"Block_{b}", index=False)
        SUMM.to_excel(xw, sheet_name="Summary", index=False)
    say("\nSaved b5_tables.xlsx and b5_summary.csv in", paths.show(paths.out_b))
    est.rep.save(paths.out_b / "part_b5_report.txt")
    return R
