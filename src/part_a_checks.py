"""
Part A, step 1: data checks for "Mental Accounting and Slack Spending" (Dunnhumby, The Complete Journey).

Reads   <data_dir>/transaction_data.csv, product.csv, hh_demographic.csv  (coupon_redempt.csv is optional)
Writes  part_a_output/part_a_report.txt               everything printed
        part_a_output/commodity_mapping_TEMPLATE.csv   one row per department x commodity, to be classified (step A3)
        part_a_output/attrition_table.csv              provisional sample-attrition table
        part_a_output/household_week_draft.csv         provisional household-week panel (used by step A2)

Run it from main.ipynb:   part_a_checks.run(paths)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import Paths
from .utils import Reporter, check_raw_data, load_table

# Provisional keyword draft of "discretionary", used ONLY for the checks in sections 4-6.
# The final classification is built in step A3 (classify_commodities.py).
DISC_PATTERN = re.compile(
    r"CANDY|SNACK|CHIP|COOKIE|CRACKER|SOFT DRINK|SODA|ICE CREAM|FROZEN DESSERT|"
    r"BEER|WINE|LIQUOR|MALT|COSMETIC|FRAGRANCE|MAGAZINE|DESSERT|CAKE|PASTRY|"
    r"CHOCOLATE|GUM|NUT|POPCORN|PRETZEL|SWEET|SPIRITS", re.I)


def run(paths: Paths) -> pd.DataFrame:
    """Run all Part A checks; returns the provisional household-week table."""
    check_raw_data(paths.data_dir)
    paths.make_dirs()
    rep = Reporter()

    tx = load_table(paths.data_dir, "transaction_data")
    prod = load_table(paths.data_dir, "product")
    demo = load_table(paths.data_dir, "hh_demographic")
    try:
        coup = load_table(paths.data_dir, "coupon_redempt")
    except FileNotFoundError:
        coup = None

    _column_names(rep, tx, prod, demo)
    baskets_per_hh = _panel_dimensions(rep, tx)
    _demographics_coverage(rep, tx, demo)
    _discount_fields(rep, tx)
    tx = _classification_prep(rep, tx, prod, paths)
    hw = _mechanical_correlation(rep, tx)
    _attrition(rep, tx, hw, demo, baskets_per_hh, paths)
    if coup is not None:
        _coupons(rep, coup)

    rep.save(paths.out_a / "part_a_report.txt")
    hw.to_csv(paths.household_week_draft, index=False)
    rep.say(f"\nReport and helper files saved in {paths.show(paths.out_a)}")
    return hw


# ---------------------------------------------------------------------------
def _column_names(rep: Reporter, tx, prod, demo) -> None:
    rep.header("0. COLUMN NAMES (use these exact names in your code and in Chapter 3)")
    rep.say("transaction_data:", list(tx.columns))
    rep.say("product:", list(prod.columns))
    rep.say("hh_demographic:", list(demo.columns))


def _panel_dimensions(rep: Reporter, tx) -> pd.Series:
    rep.header("1. PANEL DIMENSIONS (proposal says 2,500 households and 104 weeks)")
    rep.say("Rows in transaction_data:", f"{len(tx):,}")
    rep.say("Distinct households:", tx["household_key"].nunique())
    rep.say("WEEK_NO min / max:", tx["week_no"].min(), "/", tx["week_no"].max())
    rep.say("Distinct weeks observed:", tx["week_no"].nunique())
    rep.say("DAY min / max:", tx["day"].min(), "/", tx["day"].max(), "(~ days in window)")
    rep.say("Distinct baskets (trips):", f"{tx['basket_id'].nunique():,}")

    weeks_per_hh = tx.groupby("household_key")["week_no"].nunique()
    rep.say("\nActive weeks per household:")
    rep.say(weeks_per_hh.describe().round(1).to_string())
    rep.say("Share of households active in >= 50 weeks:", round((weeks_per_hh >= 50).mean(), 3))

    baskets_per_hh = tx.groupby("household_key")["basket_id"].nunique()
    rep.say("\nBaskets (visits) per household:")
    rep.say(baskets_per_hh.describe().round(1).to_string())

    # gaps: how unbalanced would a household-week panel be?
    all_weeks = np.arange(tx["week_no"].min(), tx["week_no"].max() + 1)
    n_hh = tx["household_key"].nunique()
    rep.say(f"\nFully balanced household-week panel would have {n_hh * len(all_weeks):,} rows;")
    rep.say(f"observed household-weeks with a purchase: "
            f"{tx[['household_key', 'week_no']].drop_duplicates().shape[0]:,}")
    return baskets_per_hh


def _demographics_coverage(rep: Reporter, tx, demo) -> None:
    rep.header("2. DEMOGRAPHICS COVERAGE (proposal assumes complete demographics)")
    hh_tx = set(tx["household_key"].unique())
    hh_demo = set(demo["household_key"].unique())
    rep.say("Households in hh_demographic:", len(hh_demo))
    rep.say("Of those, also present in transaction_data:", len(hh_demo & hh_tx))
    rep.say("Households in transactions WITHOUT demographics:", len(hh_tx - hh_demo))
    for col in ["income_desc", "household_size_desc", "kid_category_desc"]:
        if col in demo.columns:
            rep.say(f"\n{col}:")
            rep.say(demo[col].value_counts(dropna=False).to_string())


def _discount_fields(rep: Reporter, tx) -> None:
    """Section 3. Adds promo_savings, coupon_value and shelf_value to ``tx`` (in place)."""
    rep.header("3. DISCOUNT FIELDS: SIGNS AND DEFINITIONS")
    for c in ["sales_value", "retail_disc", "coupon_disc", "coupon_match_disc"]:
        s = tx[c]
        rep.say(f"{c:18s} min={s.min():9.2f}  max={s.max():9.2f}  "
                f"share<0={np.mean(s < 0):.3f}  share==0={np.mean(s == 0):.3f}  sum={s.sum():,.0f}")
    rep.say("\nNegative sales_value rows (returns / corrections):", int((tx["sales_value"] < 0).sum()))
    rep.say("Zero or negative quantity rows:", int((tx["quantity"] <= 0).sum()))

    # Candidate definitions. CHECK the dataset's data dictionary PDF before fixing one.
    # Discounts are stored as zero or negative numbers.
    tx["promo_savings"] = -(tx["retail_disc"] + tx["coupon_match_disc"])   # >= 0
    tx["coupon_value"] = -tx["coupon_disc"]                                  # >= 0
    tx["shelf_value"] = tx["sales_value"] + tx["promo_savings"]              # candidate pre-discount value
    rep.say("\nShare of lines with any shelf/loyalty discount:", round((tx["promo_savings"] > 0).mean(), 3))
    rep.say("Share of lines with a manufacturer coupon:", round((tx["coupon_value"] > 0).mean(), 4))
    rep.say("Share of lines with a retailer-matched coupon:", round((tx["coupon_match_disc"] < 0).mean(), 4))
    rep.say("Average implied discount rate on discounted lines:",
            round((tx.loc[tx.promo_savings > 0, "promo_savings"] /
                   tx.loc[tx.promo_savings > 0, "shelf_value"]).median(), 3))
    rep.say("\nDECISION NEEDED: confirm in the data dictionary whether sales_value is already net of")
    rep.say("retail_disc and coupon_match_disc. This fixes the definition of total basket spend.")


def _classification_prep(rep: Reporter, tx, prod, paths: Paths) -> pd.DataFrame:
    """Section 4. Merges product metadata, writes the classification template, adds a provisional keyword flag."""
    rep.header("4. PRODUCT CLASSIFICATION PREP")
    tx = tx.merge(prod[["product_id", "department", "commodity_desc"]], on="product_id", how="left")
    rep.say("Lines with no product metadata:", int(tx["department"].isna().sum()))
    rep.say("\nDepartments by share of total sales_value:")
    dept = tx.groupby("department")["sales_value"].sum().sort_values(ascending=False)
    rep.say((dept / dept.sum()).round(4).to_string())

    comm = (tx.groupby(["department", "commodity_desc"])["sales_value"].sum()
            .reset_index().sort_values("sales_value", ascending=False))
    comm["category"] = ""     # filled by step A3 (classify_commodities.py)
    comm.to_csv(paths.template, index=False)
    rep.say(f"\nSaved {paths.show(paths.template)} ({len(comm)} commodities). "
            "The 'category' column is filled in step A3.")

    tx["disc_draft"] = tx["commodity_desc"].fillna("").apply(lambda s: bool(DISC_PATTERN.search(s)))
    rep.say("\nProvisional discretionary share of total sales_value:",
            round(tx.loc[tx.disc_draft, "sales_value"].sum() / tx["sales_value"].sum(), 3))
    rep.say("Top matched commodities (sanity check the keyword draft):")
    rep.say(tx[tx.disc_draft].groupby("commodity_desc")["sales_value"].sum()
            .sort_values(ascending=False).head(25).round(0).to_string())
    return tx


def _mechanical_correlation(rep: Reporter, tx) -> pd.DataFrame:
    """Section 5. Provisional household-week panel, mechanical correlation and zeros."""
    rep.header("5. MECHANICAL CORRELATION AND ZEROS (household-week panel, provisional classification)")
    tx["disc_sales"] = np.where(tx["disc_draft"], tx["sales_value"], 0.0)
    tx["disc_promo"] = np.where(tx["disc_draft"], tx["promo_savings"], 0.0)
    tx["nec_promo"] = np.where(~tx["disc_draft"], tx["promo_savings"], 0.0)

    hw = (tx.groupby(["household_key", "week_no"])
            .agg(total_spend=("sales_value", "sum"),
                 disc_spend=("disc_sales", "sum"),
                 promo_all=("promo_savings", "sum"),
                 promo_disc=("disc_promo", "sum"),
                 promo_nec=("nec_promo", "sum"),
                 coupon=("coupon_value", "sum"),
                 n_baskets=("basket_id", "nunique"))
            .reset_index())
    hw["disc_share"] = np.where(hw.total_spend > 0, 100 * hw.disc_spend / hw.total_spend, np.nan)
    rep.say("Household-week rows:", f"{len(hw):,}")
    rep.say("Share of household-weeks with zero discretionary spend:", round((hw.disc_spend <= 0).mean(), 3))
    rep.say("Share of household-weeks with any coupon:", round((hw.coupon > 0).mean(), 3))
    rep.say("Share of household-weeks with total_spend <= 0:", round((hw.total_spend <= 0).mean(), 4))
    rep.say("\nCorrelations (these show the mechanical-correlation concern):")
    cols = ["total_spend", "disc_spend", "disc_share", "promo_all", "promo_nec", "promo_disc", "coupon"]
    rep.say(hw[cols].corr().round(2).to_string())
    rep.say("\nRead this as: if promo_all is much more correlated with disc_spend than promo_nec is,")
    rep.say("part of that link is mechanical (discounts ON discretionary items), which is why the")
    rep.say("primary regressor should be necessity-item savings.")
    return hw


def _attrition(rep: Reporter, tx, hw, demo, baskets_per_hh, paths: Paths) -> None:
    rep.header("6. SAMPLE ATTRITION TABLE")
    rows = []
    n0 = tx["household_key"].nunique()
    rows.append(("All households in transaction_data", n0, tx["basket_id"].nunique(), len(hw)))

    active = baskets_per_hh[baskets_per_hh >= 10].index
    tx1 = tx[tx["household_key"].isin(active)]
    hw1 = hw[hw["household_key"].isin(active)]
    rows.append((">= 10 store visits", len(active), tx1["basket_id"].nunique(), len(hw1)))

    complete_demo = demo.dropna(subset=["income_desc", "household_size_desc", "kid_category_desc"])
    keep = set(active) & set(complete_demo["household_key"])
    tx2 = tx[tx["household_key"].isin(keep)]
    hw2 = hw[hw["household_key"].isin(keep)]
    rows.append(("+ complete demographics", len(keep), tx2["basket_id"].nunique(), len(hw2)))

    hw3 = hw2[(hw2.total_spend > 0)]
    rows.append(("+ drop household-weeks with total spend <= 0", hw3["household_key"].nunique(), np.nan, len(hw3)))

    attr = pd.DataFrame(rows, columns=["step", "households", "baskets", "household_weeks"])
    rep.say(attr.to_string(index=False))
    attr.to_csv(paths.out_a / "attrition_table.csv", index=False)
    rep.say("\nOutliers and bulk-purchase rules are NOT applied yet; they are added in Part B (step B1).")


def _coupons(rep: Reporter, coup) -> None:
    rep.header("7. COUPON REDEMPTION TABLE")
    rep.say("Rows in coupon_redempt:", f"{len(coup):,}")
    rep.say("Households that ever redeemed:", coup["household_key"].nunique())
    rep.say("Coupon redemption is sparse; plan on a binary indicator as the main coupon variable.")
