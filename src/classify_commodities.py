"""
Part A, step 3: classify every commodity as Necessity / Discretionary / Neutral / Exclude.

Reads   part_a_output/commodity_mapping_TEMPLATE.csv      (columns: department, commodity_desc, sales_value, category)
Writes  part_a_output/commodity_mapping_CLASSIFIED.csv    and prints a coverage + sales-share summary

Run it from main.ipynb:   classify_commodities.run(paths)

Classification rule (two tests, applied in order)
-------------------------------------------------
1. Scope test:   Is it merchandise bought for household consumption or use?
                 No (fuel, coupons, deposits, prepaid cards, tickets, store supplies, service areas)  -> Exclude
2. Needs test:   Is it required to meet basic household needs (staple food, basic hygiene, cleaning,
                 health, infant care)?                                                              -> Necessity
3. Indulgence test (from the proposal's definition of Discretionary Spending): could a household forgo
                 it without compromising basic needs AND is it indulgence-oriented (consumed mainly for
                 pleasure, treat, vice, leisure, gifting or celebration)?                            -> Discretionary
                 Clear cases are tier "core"; borderline indulgence cases are tier "extended".
4. Everything else (forgoable but utilitarian, or ambiguous: pet care, hardware, coffee and tea,
   juices, vitamins, baking inputs, apparel)                                                          -> Neutral

Robustness definitions built from the tiers
-------------------------------------------
    NARROW   = Discretionary, tier "core"
    MAIN     = Discretionary, tier "core"  (proposal's list: confectionery, snacks, alcohol, desserts,
               cosmetics, magazines) plus the `snacks_other` group (borderline snack items)
    BROAD    = Discretionary, tiers "core" + "extended"
    Also use the `group` column to drop one group at a time (e.g. drop sugary_drinks or alcohol).
This is a judgement-based classification: review it, change anything you disagree with, and justify
the rule (not each item) in Appendix A.
"""
from __future__ import annotations

import pandas as pd

from .config import Paths

# ---------------------------------------------------------------------------
# 1. Scope test: whole departments excluded (service areas, store-use or non-merchandise)
# ---------------------------------------------------------------------------
DEPT_EXCLUDE = {
    "KIOSK-GAS", "MISC SALES TRAN", "MISC. TRANS.", "COUP/STR & MFG", "GM MERCH EXP",
    "CHARITABLE CONT", "POSTAL CENTER", "CNTRL/STORE SUP", "PHOTO", "VIDEO", "VIDEO RENTAL",
    "TRAVEL & LEISUR", "AUTOMOTIVE", "RX", "PHARMACY SUPPLY", "PROD-WHS SALES", "RESTAURANT",
    "ELECT &PLUMBING", "MEAT-WHSE", "DELI/SNACK BAR", "",
}

# Commodities excluded wherever they appear (non-merchandise, services, store supplies, fuel)
COMM_EXCLUDE = {
    "COUPON/MISC ITEMS", "COUPON", "COUPONS/STORE & MFG", "NO COMMODITY DESCRIPTION",
    "BOTTLE DEPOSITS", "TICKETS", "PREPAID WIRELESS&ACCESSORIES", "LONG DISTANCE CALLING CARDS",
    "IN-STORE PHOTOFINISHING", "OVERNIGHT PHOTOFINISHING", "FUEL", "PROD SUPPLIES",
    "MEAT SUPPLIES", "DELI SUPPLIES", "(CORP USE ONLY)", "MISCELLANEOUS(CORP USE ONLY)",
}

# ---------------------------------------------------------------------------
# 2-4. Rules: (category, group, tier, rationale, [commodity_desc, ...])
# ---------------------------------------------------------------------------
N, D, U = "Necessity", "Discretionary", "Neutral"
RULES = [
    # ---------------- Necessity ----------------
    (N, "meat_seafood", "", "Staple protein foods needed for household nutrition.", [
        "BEEF", "CHICKEN", "PORK", "TURKEY", "LAMB", "VEAL", "EXOTIC GAME/FOWL", "SMOKED MEATS",
        "MEAT - MISC", "MISCELLANEOUS", "RW FRESH PROCESSED MEAT", "LUNCHMEAT", "DINNER SAUSAGE",
        "HOT DOGS", "BACON", "BREAKFAST SAUSAGE/SANDWICHES", "FROZEN MEAT", "FROZEN PACKAGE MEAT",
        "MEAT - SHELF STABLE", "DELI MEATS", "CHICKEN/POULTRY", "SEAFOOD-FRESH", "SEAFOOD - FROZEN",
        "SEAFOOD - SHELF STABLE", "SEAFOOD - MISC", "FROZEN - BOXED(GROCERY)", "PKG.SEAFOOD MISC",
        "FROZEN CHICKEN", "FRZN SEAFOOD"]),
    (N, "dairy_eggs", "", "Staple dairy and eggs.", [
        "FLUID MILK PRODUCTS", "CHEESE", "CHEESES", "YOGURT", "EGGS", "BUTTER", "MARGARINES",
        "MILK BY-PRODUCTS", "MISC. DAIRY", "CANNED MILK"]),
    (N, "produce", "", "Fresh fruit and vegetables.", [
        "POTATOES", "SALAD MIX", "TOMATOES", "TROPICAL FRUIT", "APPLES", "BERRIES", "GRAPES",
        "VEGETABLES - ALL OTHERS", "CITRUS", "ONIONS", "VALUE ADDED FRUIT", "STONE FRUIT",
        "PEPPERS-ALL", "CARROTS", "MELONS", "PROCESSED", "VEGETABLES SALAD", "MUSHROOMS",
        "BROCCOLI/CAULIFLOWER", "VALUE ADDED VEGETABLES", "CORN", "PEARS", "HERBS", "SQUASH",
        "SEASONAL", "ORGANICS FRUIT & VEGETABLES"]),
    (N, "pantry_staples", "", "Staple shelf-stable and frozen foods, cooking inputs and breakfast basics.", [
        "SOUP", "DRY NOODLES/PASTA", "PASTA SAUCE", "CONDIMENTS/SAUCES", "CONDIMENTS",
        "SALD DRSNG/SNDWCH SPRD", "PNT BTR/JELLY/JAMS", "SPICES & EXTRACTS", "SHORTENING/OIL",
        "SUGARS/SWEETNERS", "FLOUR & MEALS", "DRY SAUCES/GRAVY", "HISPANIC",
        "MOLASSES/SYRUP/PANCAKE MIXS", "COLD CEREAL", "HOT CEREAL", "CEREAL/BREAKFAST",
        "DINNER MXS:DRY", "FRZN VEGETABLE/VEG DSH", "FRZN POTATOES", "FRZN FRUITS",
        "VEGETABLES - SHELF STABLE", "FRUIT - SHELF STABLE", "BEANS - CANNED GLASS & MW",
        "DRY BN/VEG/POTATO/RICE", "PICKLE/RELISH/PKLD VEG", "OLIVES", "FRZN BREAKFAST FOODS"]),
    (N, "prepared_meals", "", "Ready or convenience meals: a meal is a basic need even when convenience-priced.", [
        "DELI SPECIALTIES (RETAIL PK)"]),
    (N, "bakery_staples", "", "Bread and basic bakery staples.", [
        "BAKED BREAD/BUNS/ROLLS", "BREAD", "ROLLS", "FROZEN BREAD/DOUGH"]),
    (N, "prepared_meals", "", "Ready or convenience meals: a meal is a basic need even when convenience-priced.", [
        "FROZEN PIZZA", "FRZN MEAT/MEAT DINNERS", "HEAT/SERVE", "PREPARED FOOD", "SUSHI",
        "SANDWICHES", "SALADS/DIPS", "SALAD BAR"]),
    (N, "household_consumables", "", "Cleaning and paper consumables needed to run a household.", [
        "BATH TISSUES", "PAPER TOWELS", "FACIAL TISS/DNR NAPKIN", "FD WRAPS/BAGS/TRSH BG",
        "LAUNDRY DETERGENTS", "LAUNDRY ADDITIVES", "HOUSEHOLD CLEANG NEEDS", "DISHWASH DETERGENTS",
        "BLEACH", "BROOMS AND MOPS"]),
    (N, "personal_hygiene", "", "Basic personal hygiene and grooming.", [
        "ORAL HYGIENE PRODUCTS", "SOAP - LIQUID & BAR", "DEODORANTS", "SHAVING CARE PRODUCTS",
        "FEMININE HYGIENE", "HAIR CARE PRODUCTS", "ETHNIC PERSONAL CARE", "ADULT INCONTINENCE",
        "FAMILY PLANNING", "FOOT CARE PRODUCTS"]),
    (N, "health_otc", "", "Over-the-counter health and medical needs.", [
        "COLD AND FLU", "ANALGESICS", "ANTACIDS", "SINUS AND ALLERGY", "FIRST AID PRODUCTS",
        "LAXATIVES", "EYE AND EAR CARE PRODUCTS", "SMOKING CESSATIONS", "HOME HEALTH CARE",
        "GLASSES/VISION AIDS", "RESTRICTED DIET"]),
    (N, "infant", "", "Infant feeding and care.", [
        "INFANT FORMULA", "BABY FOODS", "DIAPERS & DISPOSABLES", "BABY HBC", "INFANT CARE PRODUCTS",
        "BABYFOOD"]),

    # ---------------- Discretionary: core ----------------
    (D, "confectionery", "core", "Sweets and candy: eaten for pleasure, forgone without affecting nutrition.", [
        "CANDY - PACKAGED", "CANDY - CHECKLANE"]),
    (D, "salty_snacks", "core", "Bagged salty snacks: indulgence food with no basic-needs role.", [
        "BAG SNACKS"]),
    (D, "sweet_bakery", "core", "Cakes, cookies, pies and sweet pastry: treat foods.", [
        "BAKED SWEET GOODS", "COOKIES/CONES", "CAKES", "BREAKFAST SWEETS", "COOKIES", "PIES",
        "SWEET GOODS & SNACKS"]),
    (D, "frozen_desserts", "core", "Ice cream, frozen novelties and frozen desserts.", [
        "ICE CREAM/MILK/SHERBTS", "FRZN NOVELTIES/WTR ICE", "FROZEN PIE/DESSERTS"]),
    (D, "sugary_drinks", "core", "Soft drinks: sugary indulgence beverage, not required for hydration.", [
        "SOFT DRINKS"]),
    (D, "alcohol", "core", "Alcoholic drinks: classic indulgence/vice good.", [
        "BEERS/ALES", "DOMESTIC WINE", "IMPORTED WINE", "MISC WINE", "LIQUOR"]),
    (D, "personal_indulgence", "core", "Cosmetics, fragrance and bath/beauty products (proposal lists cosmetics).", [
        "MAKEUP AND TREATMENT", "BATH", "FRAGRANCES", "COSMETIC ACCESSORIES"]),
    (D, "media", "core", "Magazines (proposal lists magazines).", [
        "MAGAZINE"]),

    # ---------------- Discretionary: extended (borderline indulgence) ----------------
    (D, "snacks_other", "extended", "Snack-type items whose content is mixed or borderline (crackers, nuts, popcorn, snack bars).", [
        "WAREHOUSE SNACKS", "SNACK NUTS", "POPCORN", "CHIPS&SNACKS", "PACKAGED NATURAL SNACKS",
        "SNACKS", "CRACKERS/MISC BKD FD", "CONVENIENT BRKFST/WHLSM SNACKS", "SNKS/CKYS/CRKR/CNDY"]),
    (D, "sweet_dessert_mix", "extended", "Dry dessert mixes: dessert-oriented but also a pantry item.", [
        "DRY MIX DESSERTS"]),
    (D, "sugary_drinks_other", "extended", "Sweetened drink mixes and sports drinks: similar to soft drinks, but less clear-cut.", [
        "ISOTONIC DRINKS", "PWDR/CRYSTL DRNK MX", "COCOA MIXES"]),
    (D, "tobacco", "extended", "Tobacco: vice/habit good that is forgone without compromising basic needs.", [
        "CIGARETTES", "TOBACCO OTHER", "CIGARS"]),
    (D, "media_leisure", "extended", "Leisure, entertainment and hobby goods.", [
        "BOOKSTORE", "NEWSPAPER", "AUDIO/VIDEO PRODUCTS", "FILM AND CAMERA PRODUCTS",
        "TOYS AND GAMES", "TOYS", "SPORTS MEMORABLILIA"]),
    (D, "seasonal_gifts_floral", "extended", "Gifts, celebration, seasonal and floral goods.", [
        "GREETING CARDS/WRAP/PARTY SPLY", "SPRING/SUMMER SEASONAL", "CHRISTMAS SEASONAL", "HALLOWEEN",
        "VALENTINE", "EASTER", "FALL AND WINTER SEASONAL", "FIREWORKS", "FLORAL-FRESH CUT", "ROSES",
        "FLORAL-FLOWERING PLANTS", "FLORAL-FOLIAGE PLANTS", "FLORAL BALLOONS", "FLORAL- HARD GOODS",
        "FLORAL-ACCESSORIES", "GIFT & FRUIT BASKETS", "BOUQUET (NON ROSE)", "EASTER LILY"]),

    # ---------------- Neutral ----------------
    (U, "other_beverages", "", "Habit or convenience beverages that are neither basic need nor clear indulgence.", [
        "CANNED JUICES", "REFRGRATD JUICES/DRNKS", "COFFEE", "WATER - CARBONATED/FLVRD DRINK", "TEAS",
        "FRZN JCE CONC/DRNKS", "DRY TEA/COFFEE/COCO MIX", "JUICE", "WATER", "NON-DAIRY BEVERAGES",
        "BEVERAGE", "NDAIRY/TEAS/JUICE/SOD", "SERVICE BEVERAGE"]),
    (U, "pet", "", "Pet food and supplies: forgoable by the household but a duty to the animal, not an indulgence.", [
        "DOG FOODS", "CAT FOOD", "CAT LITTER", "PET CARE SUPPLIES", "BIRD SEED"]),
    (U, "baking_misc_food", "", "Optional or mixed-use food items (baking inputs, party trays, dough, croutons).", [
        "BAKING MIXES", "BAKING NEEDS", "BAKING", "REFRGRATD DOUGH PRODUCTS", "SYRUPS/TOPPINGS",
        "MISCELLANEOUS CROUTONS", "FRZN ICE", "PARTY TRAYS", "BAKERY PARTY TRAYS",
        "COFFEE SHOP"]),
    (U, "wellness_beauty", "", "Health-food section, supplements and skin/hair items: not basic needs, not clearly indulgent.", [
        "REFRIGERATED", "FROZEN", "DRIED FRUIT", "FITNESS&DIET", "RICE CAKES", "PREPARED/PKGD FOODS",
        "BULK FOODS", "NUTS", "VITAMINS", "DIETARY AID PRODUCTS", "NATURAL VITAMINS", "NATURAL HBC",
        "MISCELLANEOUS HBC", "NEW AGE", "HAND/BODY/FACIAL PRODUCTS", "SUNTAN", "HAIR CARE ACCESSORIES",
        "PERSONAL CARE APPLIANCES"]),
    (U, "household_goods_misc", "", "Utilitarian household, hardware, apparel and general merchandise.", [
        "PAPER HOUSEWARES", "STATIONERY & SCHOOL SUPPLIES", "AIR CARE", "BATTERIES", "ELECTRICAL SUPPPLIES",
        "PLASTIC HOUSEWARES", "KITCHEN GADGETS", "COOKWARE & BAKEWARE", "CANDLES/ACCESSORIES",
        "HARDWARE SUPPLIES", "DOMESTIC GOODS", "HOSIERY/SOCKS", "AUTOMOTIVE PRODUCTS",
        "DISPOSIBLE FOILWARE", "APPAREL", "INSECTICIDES", "IRONING AND CHEMICALS",
        "LAWN AND GARDEN SHOP", "J-HOOKS", "PORTABLE ELECTRIC APPLIANCES", "HOME FURNISHINGS",
        "PROPANE", "GLASSWARE & DINNERWARE", "HOME FREEZING & CANNING SUPPLY", "CONTINUITIES",
        "SHOE CARE", "SEWING", "DOLLAR VALUE PRODUCTS", "WATCHES/CALCULATORS/LOBBY", "GARDEN CENTER",
        "CHARCOAL AND LIGHTER FLUID", "COFFEE FILTERS", "NON EDIBLE PRODUCTS", "UNKNOWN"]),
]

def norm(s):
    """Collapse repeated spaces so 'CHRISTMAS  SEASONAL' matches 'CHRISTMAS SEASONAL'."""
    return " ".join(str(s).split())


DEPT_EXCLUDE = {norm(x) for x in DEPT_EXCLUDE}
COMM_EXCLUDE = {norm(x) for x in COMM_EXCLUDE}
COMM_MAP = {}
for cat, grp, tier, why, items in RULES:
    for it in items:
        it = norm(it)
        if it in COMM_MAP and COMM_MAP[it][:3] != (cat, grp, tier):
            raise ValueError(f"Conflicting rule for commodity: {it}")
        COMM_MAP[it] = (cat, grp, tier, why)


def classify(dept, comm):
    d, c = norm(dept), norm(comm)
    if d in DEPT_EXCLUDE:
        return ("Exclude", "service_or_nonmerchandise_dept", "",
                "Department is a service area, store-use or non-merchandise category (fails the scope test).")
    if c in COMM_EXCLUDE:
        return ("Exclude", "nonmerchandise_commodity", "",
                "Fuel, coupon/misc line, deposit, prepaid card, ticket, service or store supply (fails the scope test).")
    if c in COMM_MAP:
        return COMM_MAP[c]
    return ("UNMAPPED", "", "", "No rule: add this commodity to RULES.")


def run(paths: Paths) -> pd.DataFrame:
    """Classify the template and write the classified file; returns the classified table."""
    if not paths.template.exists():
        raise FileNotFoundError(f"{paths.template.name} not found. Run step A1 (part_a_checks) first.")

    df = pd.read_csv(paths.template, keep_default_na=False)
    df["department"] = df["department"].astype(str).str.strip()
    df["commodity_desc"] = df["commodity_desc"].astype(str).str.strip()
    res = df.apply(lambda r: classify(r["department"], r["commodity_desc"]), axis=1, result_type="expand")
    res.columns = ["category", "group", "disc_tier", "rationale"]
    out = pd.concat([df.drop(columns=[c for c in ["category"] if c in df.columns]), res], axis=1)
    out["narrow_disc"] = ((out.category == "Discretionary") & (out.disc_tier == "core")).astype(int)
    out["main_disc"] = (out.narrow_disc.astype(bool) | (out.group == "snacks_other")).astype(int)
    out["broad_disc"] = (out.category == "Discretionary").astype(int)
    out.to_csv(paths.classified, index=False)

    miss = out[out.category == "UNMAPPED"]
    print(f"Rows classified: {len(out)}   unmapped: {len(miss)}")
    if len(miss):
        print(miss[["department", "commodity_desc", "sales_value"]].to_string(index=False))

    merch = out[out.category != "Exclude"]
    tot_all, tot_merch = out.sales_value.sum(), merch.sales_value.sum()
    print(f"\nShare of template sales_value excluded as non-merchandise: {1 - tot_merch / tot_all:.3f}")
    print("\nCommodity counts and share of MERCHANDISE sales (these shares are for the 360 commodities, all households):")
    s = (out.groupby("category").agg(commodities=("commodity_desc", "size"), sales=("sales_value", "sum")))
    s["share_of_merch"] = (s.sales / tot_merch).where(s.index != "Exclude")
    print(s.round(3).to_string())
    print("\nDiscretionary definitions (share of merchandise sales):")
    for col in ["narrow_disc", "main_disc", "broad_disc"]:
        print(f"  {col:12s} {out.loc[out[col] == 1, 'sales_value'].sum() / tot_merch:.3f}")
    print("\nDiscretionary groups (share of merchandise sales):")
    g = (out[out.category == "Discretionary"].groupby(["group", "disc_tier"])["sales_value"].sum() / tot_merch)
    print(g.sort_values(ascending=False).round(4).to_string())
    print(f"\nSaved {paths.show(paths.classified)}")
    return out
