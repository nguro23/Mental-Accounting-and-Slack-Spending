"""
Part A, step 2: follow-up data checks (run AFTER part_a_checks).

Reads   <data_dir>/transaction_data.csv, product.csv, hh_demographic.csv
        part_a_output/household_week_draft.csv      (written by part_a_checks)
Writes  part_a_output/part_a2_report.txt            everything printed
        part_a_output/demographics_clean.csv        demographic recodes

Definitions used here (they follow the proposal's variable definitions):
    PromotionSavings = -retail_disc                       (shelf / loyalty markdowns)
    Coupon           = -(coupon_disc + coupon_match_disc) (manufacturer + retailer-matched coupons)

Run it from main.ipynb:   part_a2_checks.run(paths)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Paths
from .utils import HH_SIZE_MAP, INCOME_MIDPOINTS, Reporter, check_raw_data, load_table


def run(paths: Paths) -> None:
    check_raw_data(paths.data_dir)
    if not paths.household_week_draft.exists():
        raise FileNotFoundError(f"{paths.household_week_draft.name} not found. Run step A1 (part_a_checks) first.")
    paths.make_dirs()
    rep = Reporter()

    tx = load_table(paths.data_dir, "transaction_data")
    prod = load_table(paths.data_dir, "product")
    demo = load_table(paths.data_dir, "hh_demographic")
    tx = tx.merge(prod[["product_id", "department", "commodity_desc"]], on="product_id", how="left")
    tx["department"] = tx["department"].fillna("").astype(str).str.strip()
    tx["savings"] = -tx["retail_disc"]
    tx["coupon_amt"] = -(tx["coupon_disc"] + tx["coupon_match_disc"])

    _a_net_or_gross(rep, tx)
    _b_positive_discounts(rep, tx)
    _c_nonpositive_quantity(rep, tx)
    _d_nonmerchandise_departments(rep, tx)

    hw = pd.read_csv(paths.household_week_draft)
    demo_hh = set(demo["household_key"])
    hw["in_demo"] = hw["household_key"].isin(demo_hh)
    _e_demographic_subset(rep, hw, demo_hh)
    _f_activity_filter(rep, hw, demo_hh)
    _g_demographic_recodes(rep, demo, paths)
    _h_coupon_sparseness(rep, hw)

    rep.save(paths.out_a / "part_a2_report.txt")
    rep.say(f"\nSaved {paths.show(paths.out_a / 'part_a2_report.txt')} and "
            f"{paths.show(paths.out_a / 'demographics_clean.csv')}")


# ---------------------------------------------------------------------------
def _a_net_or_gross(rep: Reporter, tx) -> None:
    rep.header("A. IS sales_value NET OR GROSS OF retail_disc?  (shelf-price stability test)")
    # Idea: the undiscounted shelf price of a product in a store-week should be the same
    # across lines. Whichever candidate unit price varies LESS within product-store-week
    # (when the group mixes discounted and undiscounted lines) is the undiscounted price.
    d = tx[(tx.quantity > 0) & (tx.sales_value > 0) & (tx.retail_disc <= 0)].copy()
    top = d["product_id"].value_counts().head(300).index          # speed: most frequent products
    d = d[d["product_id"].isin(top)]
    d["u_as_is"] = d["sales_value"] / d["quantity"]
    d["u_plus_disc"] = (d["sales_value"] + d["savings"]) / d["quantity"]
    d["disc_flag"] = (d["savings"] > 0).astype(int)
    agg = (d.groupby(["product_id", "store_id", "week_no"])
             .agg(n=("disc_flag", "size"), nd=("disc_flag", "sum"),
                  m1=("u_as_is", "mean"), s1=("u_as_is", "std"),
                  m2=("u_plus_disc", "mean"), s2=("u_plus_disc", "std"))
             .reset_index())
    agg = agg[(agg.n >= 3) & (agg.nd >= 1) & (agg.nd < agg.n)]
    agg["cv_as_is"] = agg.s1 / agg.m1
    agg["cv_plus_disc"] = agg.s2 / agg.m2
    rep.say("Product-store-weeks with mixed discounted / undiscounted lines:", f"{len(agg):,}")
    rep.say("Median coefficient of variation, sales_value/quantity:           ", round(agg.cv_as_is.median(), 4))
    rep.say("Median coefficient of variation, (sales_value+savings)/quantity: ", round(agg.cv_plus_disc.median(), 4))
    if len(agg):
        if agg.cv_plus_disc.median() < agg.cv_as_is.median():
            rep.say("=> Adding savings back makes prices MORE stable: sales_value looks NET of retail_disc.")
            rep.say("   Shelf (pre-discount) value = sales_value + savings. Customer paid ~ sales_value.")
        else:
            rep.say("=> sales_value alone is MORE stable: it looks GROSS (pre-discount).")
            rep.say("   Customer paid ~ sales_value - savings. Check the dictionary to confirm.")
    rep.say("Always confirm against the dunnhumby data dictionary / user guide as well.")


def _b_positive_discounts(rep: Reporter, tx) -> None:
    rep.header("B. POSITIVE retail_disc VALUES (sign assumption fails for some lines)")
    pos = tx[tx.retail_disc > 0]
    rep.say("Lines with retail_disc > 0:", f"{len(pos):,}", f"({len(pos) / len(tx):.4%} of lines)")
    rep.say("Sum of positive retail_disc: ", round(pos.retail_disc.sum(), 2))
    rep.say("Share of those lines with quantity <= 0:", round((pos.quantity <= 0).mean(), 3))
    rep.say("Share of those lines with sales_value == 0:", round((pos.sales_value == 0).mean(), 3))
    if len(pos):
        rep.say("\nBy department (top 8):")
        rep.say(pos.groupby("department")["retail_disc"].agg(["size", "sum"])
                .sort_values("size", ascending=False).head(8).round(2).to_string())
        rep.say("\nSample rows:")
        rep.say(pos[["household_key", "department", "quantity", "sales_value", "retail_disc",
                     "coupon_disc"]].head(8).to_string(index=False))
    rep.say("\nTreatment options: (i) floor savings at zero, (ii) keep as signed reversals,")
    rep.say("(iii) drop those lines. Report which one you used, and test sensitivity.")


def _c_nonpositive_quantity(rep: Reporter, tx) -> None:
    rep.header("C. LINES WITH quantity <= 0")
    q = tx[tx.quantity <= 0]
    rep.say("quantity == 0:", f"{int((tx.quantity == 0).sum()):,}", "  quantity < 0:", f"{int((tx.quantity < 0).sum()):,}")
    rep.say("Share of all lines:", round(len(q) / len(tx), 4))
    rep.say("sales_value on these lines:")
    rep.say(q["sales_value"].describe().round(2).to_string())
    rep.say("Share of total sales_value on these lines:", round(q.sales_value.sum() / tx.sales_value.sum(), 4))
    rep.say("\nTop departments:")
    rep.say(q.groupby("department").agg(lines=("quantity", "size"), sales=("sales_value", "sum"))
            .sort_values("lines", ascending=False).head(8).round(0).to_string())


def _d_nonmerchandise_departments(rep: Reporter, tx) -> None:
    rep.header("D. NON-MERCHANDISE / OUT-OF-SCOPE DEPARTMENTS (candidates to exclude)")
    dep = tx.groupby("department").agg(lines=("quantity", "size"), sales=("sales_value", "sum"),
                                       savings=("savings", "sum"))
    dep["sales_share"] = dep.sales / dep.sales.sum()
    dep["savings_share"] = dep.savings / dep.savings.sum()
    cand = ["KIOSK-GAS", "MISC SALES TRAN", "MISC. TRANS.", "COUP/STR & MFG", "GM MERCH EXP",
            "CHARITABLE CONT", "POSTAL CENTER", "CNTRL/STORE SUP", "PHOTO", "VIDEO", "VIDEO RENTAL",
            "TRAVEL & LEISUR", "AUTOMOTIVE", "RX", "PHARMACY SUPPLY", "PROD-WHS SALES",
            "RESTAURANT", "GARDEN CENTER", "FLORAL", ""]
    sel = dep.reindex([c for c in cand if c in dep.index])
    rep.say(sel.round(4).to_string())
    rep.say("\nCombined share of sales_value in these departments:", round(sel.sales_share.sum(), 4))
    rep.say("Combined share of retail savings in these departments:", round(sel.savings_share.sum(), 4))
    if "KIOSK-GAS" in tx.department.values:
        rep.say("\nKIOSK-GAS commodities:")
        rep.say(tx[tx.department == "KIOSK-GAS"].groupby("commodity_desc")["sales_value"].sum()
                .sort_values(ascending=False).head(6).round(0).to_string())
    rep.say("\nFuel is neither necessity nor discretionary in your framework. If it stays in total spend,")
    rep.say("it contaminates the DiscretionaryShare denominator. Recommended: merchandise-only total.")


def _e_demographic_subset(rep: Reporter, hw, demo_hh) -> None:
    rep.header("E. IS THE DEMOGRAPHIC SUBSET (801 hh) DIFFERENT FROM THE OTHER 1,699?")
    hh = (hw.groupby("household_key")
            .agg(active_weeks=("week_no", "nunique"),
                 baskets=("n_baskets", "sum"),
                 mean_weekly_spend=("total_spend", "mean"),
                 mean_disc_share=("disc_share", "mean"),
                 mean_promo_nec=("promo_nec", "mean"),
                 share_weeks_coupon=("coupon", lambda s: (s > 0).mean()))
            .reset_index())
    hh["in_demo"] = hh["household_key"].isin(demo_hh)
    rows = []
    for c in ["active_weeks", "baskets", "mean_weekly_spend", "mean_disc_share",
              "mean_promo_nec", "share_weeks_coupon"]:
        a, b = hh.loc[hh.in_demo, c], hh.loc[~hh.in_demo, c]
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append((c, round(a.mean(), 2), round(b.mean(), 2), round((a.mean() - b.mean()) / sd, 2)))
    rep.say(pd.DataFrame(rows, columns=["variable", "with_demographics", "without", "std_diff"]).to_string(index=False))
    rep.say("\nRule of thumb: |std_diff| > 0.25 suggests a meaningfully different group.")


def _f_activity_filter(rep: Reporter, hw, demo_hh) -> None:
    rep.header("F. ACTIVITY FILTER: visits vs active weeks, entry/exit pattern")
    act = hw.groupby("household_key").agg(active_weeks=("week_no", "nunique"),
                                          baskets=("n_baskets", "sum"),
                                          wk_first=("week_no", "min"),
                                          wk_last=("week_no", "max"))
    act["span"] = act.wk_last - act.wk_first + 1
    act["fill"] = act.active_weeks / act.span
    act["in_demo"] = act.index.isin(demo_hh)
    rep.say("Households with >= 10 baskets but < 10 active weeks:",
            int(((act.baskets >= 10) & (act.active_weeks < 10)).sum()))
    for k in [5, 10, 20, 26, 52]:
        rep.say(f"Households with >= {k:>2} active weeks: all={int((act.active_weeks >= k).sum()):>5}   "
                f"with demographics={int(((act.active_weeks >= k) & act.in_demo).sum()):>4}")
    rep.say("\nMedian share of weeks active within own first-to-last-week span:", round(act.fill.median(), 3))
    rep.say("Share of households whose first purchase is after week 10:", round((act.wk_first > 10).mean(), 3))
    rep.say("Share of households whose last purchase is before week 93:", round((act.wk_last < 93).mean(), 3))
    rep.say("(High fill within span => zero-filling inside the span is defensible as a robustness check;")
    rep.say(" low fill => households shop sporadically and missing weeks may be real non-shopping.)")


def _g_demographic_recodes(rep: Reporter, demo, paths: Paths) -> None:
    rep.header("G. DEMOGRAPHIC RECODES")
    dm = demo.copy()
    dm["income_k_mid"] = dm["income_desc"].map(INCOME_MIDPOINTS)
    dm["hh_size_num"] = dm["household_size_desc"].astype(str).map(HH_SIZE_MAP)
    dm["has_kids"] = (dm["kid_category_desc"] != "None/Unknown").astype(int)
    rep.say("Unmapped income bands:", int(dm.income_k_mid.isna().sum()),
            "  unmapped household sizes:", int(dm.hh_size_num.isna().sum()))
    rep.say("has_kids (1 = 1, 2 or 3+ children; 0 = None/Unknown):")
    rep.say(dm.has_kids.value_counts().to_string())
    rep.say("\nNOTE: 'None/Unknown' mixes households with no children and households whose status is")
    rep.say("unknown, so has_kids = 0 is contaminated. State this as a limitation.")
    rep.say("NOTE: household size '5+' is top-coded at 5; income midpoints are an assumption (250K+ -> 275K).")
    dm.to_csv(paths.out_a / "demographics_clean.csv", index=False)


def _h_coupon_sparseness(rep: Reporter, hw) -> None:
    rep.header("H. COUPON SPARSENESS AND WITHIN-HOUSEHOLD VARIATION (demographic subset)")
    sub = hw[hw.in_demo].copy()
    rep.say("Household-weeks:", f"{len(sub):,}", " households:", sub.household_key.nunique())
    rep.say("Share of household-weeks with any coupon:", round((sub.coupon > 0).mean(), 3))
    cw = sub.groupby("household_key")["coupon"].apply(lambda s: int((s > 0).sum()))
    rep.say("Households with >= 1 coupon week:", int((cw >= 1).sum()),
            "  >= 3 coupon weeks:", int((cw >= 3).sum()), "  >= 10 coupon weeks:", int((cw >= 10).sum()))
    rep.say("\nWithin-household share of total variance (higher = more variation FE can use):")
    for c in ["disc_spend", "disc_share", "total_spend", "promo_nec", "promo_all", "coupon"]:
        x = sub[c].astype(float)
        ok = x.notna()
        xw = x[ok] - x[ok].groupby(sub.loc[ok, "household_key"]).transform("mean")
        rep.say(f"  {c:12s} within/total variance = {xw.var() / x[ok].var():.3f}")
