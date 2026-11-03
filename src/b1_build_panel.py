"""
Part B, step 1: build the analysis panels for "Mental Accounting and Slack Spending".

Reads   <data_dir>/transaction_data.csv, product.csv, hh_demographic.csv
        part_a_output/commodity_mapping_CLASSIFIED.csv                (from step A3)
Writes  part_b_output/hw_all.csv              household-week panel, all households passing the activity + outlier filters
        part_b_output/hw_main.csv             MAIN analytical sample: the same, restricted to households with demographics
        part_b_output/trips_main.csv          household-trip (basket) version of the main sample (robustness check)
        part_b_output/attrition_table.csv     sample attrition table (Chapter 4.3)
        part_b_output/variable_dictionary.csv definition of every column (Chapters 3 and 4.3)
        part_b_output/part_b1_report.txt      everything printed

Run it from main.ipynb:   b1_build_panel.run(paths, min_active_weeks=10, outlier_q=0.999)

Definitions (match Chapter 3):
    Merchandise lines        = lines not classed Exclude, with quantity > 0
    total_spend              = sum of SALES_VALUE over merchandise lines
    PromotionSavings         = sum of max(-RETAIL_DISC, 0); primary = on Necessity items (promo_nec),
                               secondary = on all merchandise (promo_all)
    Coupon                   = sum of max(-(COUPON_DISC + COUPON_MATCH_DISC), 0); coupon_any = 1 if > 0
    DiscretionarySpending    = spend on the MAIN discretionary definition (core + borderline snacks)
    DiscretionaryShare       = 100 * DiscretionarySpending / total_spend
    Week index               = WEEK_NO (1-102); unit of analysis = household-week with total_spend > 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Paths
from .utils import HH_SIZE_MAP, INCOME_MIDPOINTS, Reporter, check_raw_data, load_table

VALID_CATEGORIES = {"Necessity", "Discretionary", "Neutral", "Exclude"}
CLASSIFICATION_COLUMNS = ["department", "commodity_desc", "category", "group", "disc_tier",
                          "narrow_disc", "main_disc", "broad_disc"]

# What each household-week column is built from (name = (line-level column, aggregation))
AGG_SPEC = dict(
    total_spend=("spend", "sum"),
    nec_spend=("spend_nec", "sum"),
    disc_spend=("spend_disc_main", "sum"),
    disc_narrow_spend=("spend_disc_narrow", "sum"),
    disc_broad_spend=("spend_disc_broad", "sum"),
    disc_xsugar_spend=("spend_disc_xsugar", "sum"),
    disc_xalc_spend=("spend_disc_xalc", "sum"),
    neutral_spend=("spend_neutral", "sum"),
    promo_all=("sav", "sum"),
    promo_nec=("sav_nec", "sum"),
    promo_disc=("sav_disc_main", "sum"),
    promo_neutral=("sav_neutral", "sum"),
    coupon=("cpn", "sum"),
    coupon_nec=("cpn_nec", "sum"),
    coupon_disc=("cpn_disc_main", "sum"),
    n_lines=("product_id", "size"),
    n_baskets=("basket_id", "nunique"),
)

VARIABLE_DICTIONARY = [
    ("household_key / week_no", "Household id and week (1-102); the panel keys. Time fixed effects use week_no."),
    ("total_spend", "Sum of SALES_VALUE over merchandise lines in the household-week (excludes fuel, coupon/misc, services)."),
    ("nec_spend", "Spend on Necessity commodities."),
    ("disc_spend", "DiscretionarySpending (MAIN definition: core hedonic + borderline snacks)."),
    ("disc_narrow_spend / disc_broad_spend", "Discretionary spend under the narrow (core only) and broad (core + extended) definitions."),
    ("disc_xsugar_spend / disc_xalc_spend", "Main discretionary spend excluding sugary drinks / alcohol (robustness)."),
    ("neutral_spend", "Spend on Neutral commodities (falsification outcome)."),
    ("disc_share", "DiscretionaryShare = 100 * disc_spend / total_spend (also _narrow_, _broad_ versions)."),
    ("nec_share / neutral_share", "Necessity and Neutral shares of total_spend."),
    ("promo_nec", "PromotionSavings (PRIMARY): sum of max(-RETAIL_DISC, 0) on Necessity items."),
    ("promo_all", "PromotionSavings (secondary): same, on all merchandise."),
    ("promo_disc / promo_neutral", "Same, on main-discretionary / Neutral items (diagnostics; mechanically linked to disc_spend)."),
    ("promo_nec_rate", "Necessity markdown depth: 100 * promo_nec / (nec_spend + promo_nec); NaN if no necessity spend."),
    ("nec_spend_shelf", "Necessity spend before markdowns (nec_spend + promo_nec); alternative control."),
    ("coupon / coupon_any", "Coupon: sum of -(COUPON_DISC + COUPON_MATCH_DISC) (>=0); coupon_any = 1 if > 0."),
    ("spend_net_coupon", "total_spend minus coupon value (sensitivity if SALES_VALUE is gross of coupons)."),
    ("n_baskets / n_lines", "Trips and merchandise lines in the household-week."),
    ("income_k_mid / income_band", "Income band midpoint ($000, assumption: 250K+ -> 275) and ordinal band 1-12."),
    ("hh_size_num", "Household size (5+ top-coded at 5)."),
    ("has_kids", "1 if KID_CATEGORY_DESC is 1, 2 or 3+; 0 if 'None/Unknown' (mixes none and unknown)."),
    ("has_demo", "1 if the household has a record in hh_demographic (all 1 in hw_main)."),
]


def run(paths: Paths, min_active_weeks: int = 10, outlier_q: float = 0.999) -> dict[str, pd.DataFrame]:
    """Build the panels. ``min_active_weeks``: drop households observed in fewer weeks;
    ``outlier_q``: drop household-weeks with total_spend above this quantile (bulk purchases)."""
    check_raw_data(paths.data_dir)
    if not paths.classified.exists():
        raise FileNotFoundError(f"{paths.classified.name} not found. Run step A3 (classify_commodities) first.")
    paths.make_dirs()
    rep = Reporter()

    tx, demo, raw_counts = _load_and_merge(rep, paths)
    txm = _clean_lines(rep, tx)
    hw = _aggregate_household_weeks(rep, txm)
    hw_all, hw_main, dm = _filter_and_attrition(rep, hw, demo, raw_counts, paths, min_active_weeks, outlier_q)
    trips = _build_trips(rep, txm, hw_main, dm)
    _sanity_checks(rep, hw_main)
    _save(rep, paths, hw_all, hw_main, trips)
    return {"hw_all": hw_all, "hw_main": hw_main, "trips_main": trips}


# ---------------------------------------------------------------------------
def _load_and_merge(rep: Reporter, paths: Paths):
    rep.header("1. LOAD AND MERGE (transactions + product + classification)")
    tx = load_table(paths.data_dir, "transaction_data")
    prod = load_table(paths.data_dir, "product")
    demo = load_table(paths.data_dir, "hh_demographic")
    cls = pd.read_csv(paths.classified, keep_default_na=False)

    for df in (prod, cls):
        df["department"] = df["department"].fillna("").astype(str).str.strip()
        df["commodity_desc"] = df["commodity_desc"].fillna("").astype(str).str.strip()

    tx = tx.merge(prod[["product_id", "department", "commodity_desc"]], on="product_id", how="left")
    tx["department"] = tx["department"].fillna("").astype(str).str.strip()
    tx["commodity_desc"] = tx["commodity_desc"].fillna("").astype(str).str.strip()
    tx = tx.merge(cls[CLASSIFICATION_COLUMNS].drop_duplicates(["department", "commodity_desc"]),
                  on=["department", "commodity_desc"], how="left")

    unm = tx[tx["category"].isna() | ~tx["category"].isin(VALID_CATEGORIES)]
    rep.say("Transaction lines:", f"{len(tx):,}", "  lines with no valid classification:", f"{len(unm):,}")
    if len(unm):
        u = (unm.assign(category=unm["category"].fillna("(no match)"))
                .groupby(["department", "commodity_desc", "category"])["sales_value"]
                .agg(lines="size", sales="sum").sort_values("sales", ascending=False))
        rep.say(u.head(40).round(0).to_string())
        raise RuntimeError("\nThe commodities above are 'UNMAPPED' or missing in the classification file. Add rules for "
                           "them in classify_commodities.RULES (or set their category by hand in "
                           "commodity_mapping_CLASSIFIED.csv), then re-run steps A3 and B1.")

    raw_counts = dict(households=tx["household_key"].nunique(), baskets=tx["basket_id"].nunique(),
                      household_weeks=tx[["household_key", "week_no"]].drop_duplicates().shape[0])
    return tx, demo, raw_counts


def _clean_lines(rep: Reporter, tx: pd.DataFrame) -> pd.DataFrame:
    """Section 2: drop non-merchandise and quantity <= 0 lines, build line-level spend / savings / coupon columns."""
    rep.header("2. LINE-LEVEL CLEANING")
    s0 = tx["sales_value"].sum()
    nonmerch = tx["category"].eq("Exclude")
    rep.say(f"Non-merchandise lines (category = Exclude): {int(nonmerch.sum()):,} "
            f"({nonmerch.mean():.1%} of lines, {tx.loc[nonmerch, 'sales_value'].sum() / s0:.1%} of sales_value)")
    m = tx[~nonmerch]
    zq = m["quantity"] <= 0
    rep.say(f"Merchandise lines with quantity <= 0: {int(zq.sum()):,} "
            f"(sales_value on them: {m.loc[zq, 'sales_value'].sum():,.2f}) -> dropped")
    txm = m[~zq].copy()
    rep.say("Merchandise lines kept:", f"{len(txm):,}")

    pos = txm["retail_disc"] > 0
    rep.say(f"Lines with positive RETAIL_DISC: {int(pos.sum())} (total {txm.loc[pos, 'retail_disc'].sum():.2f}) "
            "-> savings floored at zero")
    txm["spend"] = txm["sales_value"]
    txm["sav"] = (-txm["retail_disc"]).clip(lower=0)
    txm["cpn"] = (-(txm["coupon_disc"] + txm["coupon_match_disc"])).clip(lower=0)

    cat = txm["category"]
    masks = {
        "nec": cat.eq("Necessity"),
        "neutral": cat.eq("Neutral"),
        "disc_main": txm["main_disc"].eq(1),
        "disc_narrow": txm["narrow_disc"].eq(1),
        "disc_broad": txm["broad_disc"].eq(1),
        "disc_xsugar": txm["main_disc"].eq(1) & txm["group"].ne("sugary_drinks"),
        "disc_xalc": txm["main_disc"].eq(1) & txm["group"].ne("alcohol"),
    }
    for k, mk in masks.items():
        txm["spend_" + k] = np.where(mk, txm["spend"], 0.0)
    for k in ["nec", "neutral", "disc_main"]:
        txm["sav_" + k] = np.where(masks[k], txm["sav"], 0.0)
    for k in ["nec", "disc_main"]:                      # coupons are item-specific: split them by item category
        txm["cpn_" + k] = np.where(masks[k], txm["cpn"], 0.0)
    return txm


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Shares, markdown depth and other variables derived from the aggregated columns (household-week or trip)."""
    df = df[df["total_spend"] > 0].copy()
    t = df["total_spend"]
    df["disc_share"] = 100 * df["disc_spend"] / t
    df["disc_narrow_share"] = 100 * df["disc_narrow_spend"] / t
    df["disc_broad_share"] = 100 * df["disc_broad_spend"] / t
    df["nec_share"] = 100 * df["nec_spend"] / t
    df["neutral_share"] = 100 * df["neutral_spend"] / t
    df["coupon_any"] = (df["coupon"] > 0).astype(int)
    df["nec_spend_shelf"] = df["nec_spend"] + df["promo_nec"]          # necessity spend before markdowns
    df["promo_nec_rate"] = np.where(df["nec_spend_shelf"] > 0,
                                    100 * df["promo_nec"] / df["nec_spend_shelf"], np.nan)
    df["spend_net_coupon"] = (df["total_spend"] - df["coupon"]).clip(lower=0)   # sensitivity only
    return df


def _aggregate_household_weeks(rep: Reporter, txm: pd.DataFrame) -> pd.DataFrame:
    rep.header("3. HOUSEHOLD-WEEK AGGREGATION")
    hw = _derive(txm.groupby(["household_key", "week_no"]).agg(**AGG_SPEC).reset_index())
    rep.say("Household-weeks with merchandise spend > 0:", f"{len(hw):,}", " households:", hw["household_key"].nunique())

    # identity check: the three categories must add up to total merchandise spend
    gap = (hw["nec_spend"] + hw["disc_broad_spend"] + hw["neutral_spend"] - hw["total_spend"]).abs().max()
    rep.say(f"Check: Necessity + Discretionary(broad) + Neutral = total_spend, max abs gap = {gap:.2e}")
    assert gap < 1e-6, "category spend does not add up to total spend: check the classification file"
    rep.say("Check: main <= broad discretionary spend:", bool((hw["disc_spend"] <= hw["disc_broad_spend"] + 1e-9).all()))
    return hw


def _demographics(demo: pd.DataFrame) -> pd.DataFrame:
    """Recode the demographic record (income midpoint and band, household size, children)."""
    inc_order = list(INCOME_MIDPOINTS.keys())
    dm = demo[["household_key", "income_desc", "household_size_desc", "kid_category_desc"]].dropna().copy()
    dm["income_k_mid"] = dm["income_desc"].map(INCOME_MIDPOINTS)
    dm["income_band"] = dm["income_desc"].map({k: i + 1 for i, k in enumerate(inc_order)})
    dm["hh_size_num"] = dm["household_size_desc"].astype(str).map(HH_SIZE_MAP)
    dm["has_kids"] = (dm["kid_category_desc"] != "None/Unknown").astype(int)
    dm = dm.dropna(subset=["income_k_mid", "hh_size_num"])
    dm["has_demo"] = 1
    return dm


def _filter_and_attrition(rep: Reporter, hw, demo, raw_counts, paths: Paths, min_active_weeks, outlier_q):
    rep.header("4. SAMPLE FILTERS AND ATTRITION")
    rows = [("All households in transaction_data", raw_counts["households"], raw_counts["household_weeks"],
             raw_counts["baskets"])]
    rows.append(("Merchandise lines only (non-merchandise and quantity <= 0 removed), total_spend > 0",
                 hw["household_key"].nunique(), len(hw), int(hw["n_baskets"].sum())))

    wk = hw.groupby("household_key")["week_no"].nunique()
    act = wk[wk >= min_active_weeks].index
    hw2 = hw[hw["household_key"].isin(act)]
    rows.append((f">= {min_active_weeks} active weeks", len(act), len(hw2), int(hw2["n_baskets"].sum())))

    cut = hw2["total_spend"].quantile(outlier_q)
    hw3 = hw2[hw2["total_spend"] <= cut].copy()
    rows.append((f"Drop bulk-purchase weeks (total_spend > p{outlier_q * 100:g} = ${cut:,.0f})",
                 hw3["household_key"].nunique(), len(hw3), int(hw3["n_baskets"].sum())))

    dm = _demographics(demo)
    hw_all = hw3.merge(dm, on="household_key", how="left")
    hw_all["has_demo"] = hw_all["has_demo"].fillna(0).astype(int)
    hw_main = hw_all[hw_all["has_demo"] == 1].copy()
    rows.append(("MAIN SAMPLE: households with a demographic record", hw_main["household_key"].nunique(),
                 len(hw_main), int(hw_main["n_baskets"].sum())))

    attr = pd.DataFrame(rows, columns=["step", "households", "household_weeks", "baskets"])
    rep.say(attr.to_string(index=False))
    attr.to_csv(paths.out_b / "attrition_table.csv", index=False)
    rep.say(f"\nActive weeks per household in MAIN sample: min {hw_main.groupby('household_key').size().min()}, "
            f"median {hw_main.groupby('household_key').size().median():.0f}")
    return hw_all, hw_main, dm


def _build_trips(rep: Reporter, txm, hw_main, dm) -> pd.DataFrame:
    """Section 5: household-trip (basket) version of the main sample, for robustness."""
    rep.header("5. HOUSEHOLD-TRIP (BASKET) VERSION FOR ROBUSTNESS")
    main_hh = set(hw_main["household_key"])
    tm = txm[txm["household_key"].isin(main_hh)]
    spec_trip = dict(AGG_SPEC)
    spec_trip.pop("n_baskets")
    trips = tm.groupby(["household_key", "basket_id"]).agg(week_no=("week_no", "first"), day=("day", "min"),
                                                           **spec_trip).reset_index()
    trips = _derive(trips)
    trips = trips.merge(hw_main[["household_key", "week_no"]], on=["household_key", "week_no"], how="inner")
    trips = trips.merge(dm.drop(columns=["income_desc", "household_size_desc", "kid_category_desc"]),
                        on="household_key", how="left")
    rep.say("Trips in main sample:", f"{len(trips):,}", " households:", trips["household_key"].nunique())
    return trips


def _sanity_checks(rep: Reporter, S: pd.DataFrame) -> None:
    rep.header("6. DESCRIPTIVE SANITY CHECKS (main sample, household-week)")
    rep.say("Mean weekly total_spend:", round(S.total_spend.mean(), 2), "  median:", round(S.total_spend.median(), 2))
    rep.say("Spend by category (share of total merchandise spend):",
            {c: round(float(S[c].sum() / S.total_spend.sum()), 3) for c in
             ["nec_spend", "disc_spend", "disc_narrow_spend", "disc_broad_spend", "neutral_spend"]})
    rep.say("Household-weeks with zero discretionary (main) spend:", round((S.disc_spend <= 0).mean(), 3))
    rep.say("Household-weeks with any coupon:", round(S.coupon_any.mean(), 3))
    rep.say("Household-weeks with zero necessity savings:", round((S.promo_nec <= 0).mean(), 3))
    rep.say("Mean necessity-item markdown depth (% of shelf value), spend-weighted:",
            round(100 * S.promo_nec.sum() / S.nec_spend_shelf.sum(), 2))
    cols = ["disc_spend", "disc_share", "total_spend", "nec_spend", "promo_nec", "promo_all", "promo_disc", "coupon"]
    rep.say("\nSummary statistics:")
    rep.say(S[cols].describe(percentiles=[.01, .25, .5, .75, .99]).T.round(2).to_string())
    rep.say("\nCorrelations (main sample):")
    rep.say(S[cols].corr().round(2).to_string())

    rep.say("\nWithin-household share of variance:")
    for c in ["disc_spend", "disc_share", "total_spend", "nec_spend", "promo_nec", "coupon"]:
        x = S[c].astype(float)
        xw = x - x.groupby(S["household_key"]).transform("mean")
        rep.say(f"  {c:12s} {xw.var() / x.var():.3f}")


def _save(rep: Reporter, paths: Paths, hw_all, hw_main, trips) -> None:
    rep.header("7. SAVE")
    pd.DataFrame(VARIABLE_DICTIONARY, columns=["variable", "definition"]).to_csv(
        paths.out_b / "variable_dictionary.csv", index=False)
    hw_all.to_csv(paths.hw_all, index=False)
    hw_main.to_csv(paths.hw_main, index=False)
    trips.to_csv(paths.trips_main, index=False)
    rep.say("Saved hw_all.csv, hw_main.csv, trips_main.csv, attrition_table.csv, variable_dictionary.csv in",
            paths.show(paths.out_b))
    rep.save(paths.out_b / "part_b1_report.txt")
