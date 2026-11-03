# Mental Accounting and Slack Spending

**Evidence of Discretionary Reallocation of Promotional Savings from the Dunnhumby Dataset**

Replication code for the undergraduate thesis by Mwangi David Nguro (BBS Financial Economics, Strathmore University). The thesis asks whether households treat promotional savings on necessity goods as a windfall and move them into discretionary purchases, as mental accounting predicts. It tests this on household-week panel data built from *Dunnhumby: The Complete Journey*.

Everything runs from one notebook, [`main.ipynb`](main.ipynb). It reproduces the commodity classification (thesis Appendix A) and every table, figure and report used in Chapter 4, starting from the raw Dunnhumby files.

## Repository layout

```
Mental-Accounting-and-Slack-Spending/
├── main.ipynb              the only file you need to run
├── requirements.txt        Python packages
├── src/                    one module per step of the pipeline (called by main.ipynb)
│   ├── config.py           folder layout
│   ├── utils.py            helpers shared by the steps
│   ├── part_a_checks.py, part_a2_checks.py, classify_commodities.py          Part A
│   └── b1_build_panel.py, b2_diagnostics.py, b3_descriptives.py,
│       b4_models.py, b5_robustness.py, b5_forest_figure.py                    Part B
├── data/raw/               put the Dunnhumby CSV files here (not included; see below)
├── part_a_output/          Part A results: reports, classification, helper tables
└── part_b_output/          Part B results: panels, diagnostics, tables, figures, reports
```

## Getting the data

The dataset is **not included** in this repository. Download *The Complete Journey* from either source:

* dunnhumby: <https://www.dunnhumby.com/source-files/>
* Kaggle mirror: <https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey>

Unzip it and copy the CSV files into `data/raw/`. The pipeline reads these files:

| File | Needed | Used for |
|---|---|---|
| `transaction_data.csv` | yes | purchases, prices, discounts and coupons |
| `product.csv` | yes | department and commodity of each product |
| `hh_demographic.csv` | yes | income, household size and children (801 households) |
| `coupon_redempt.csv` | optional | one descriptive check in step A1 |

The other files in the download (`campaign_desc.csv`, `campaign_table.csv`, `causal_data.csv`, `coupon.csv`) are not used.

If the files are somewhere else, set `DATA_DIR` in the first code cell of `main.ipynb` (for example `DATA_DIR = "../dunnhumby-the-complete-journey"`), or set the environment variable `DUNNHUMBY_DATA_DIR`.

## How to run

1. Install Python 3.9 or newer, then the packages: `pip install -r requirements.txt`
2. Put the data in `data/raw/` as described above.
3. Start Jupyter from the repository folder (`jupyter lab`), open `main.ipynb` and choose **Kernel ▸ Restart & Run All**.

The data contain about 2.6 million transaction lines, so the full run takes a while. The pipeline contains no random numbers, so repeated runs give the same results. The first code cells print the software versions used.

Each step prints a report and saves the same text as `part_*_report.txt`; compare these with the files in this repository to confirm that your run reproduces the thesis numbers.

### Settings

| Setting | Where | Default | Meaning |
|---|---|---|---|
| `DATA_DIR` | notebook, setup cell | `data/raw` | folder with the Dunnhumby CSV files |
| `min_active_weeks` | notebook, step B1 | 10 | households observed in fewer weeks are dropped |
| `outlier_q` | notebook, step B1 | 0.999 | household-weeks with total spend above this quantile (bulk purchases) are dropped |

The classification of commodities is defined by the rules in `src/classify_commodities.py`. To change the category of a commodity group, edit its rule there and re-run steps A3 and B1 to B5.

## What each step produces

Run the steps in this order (the notebook does). The thesis sections come from Appendix B.

| Step | Module in `src/` | Main outputs | Thesis |
|---|---|---|---|
| A1, A2 | `part_a_checks`, `part_a2_checks` | `part_a_report.txt`, `part_a2_report.txt`, `commodity_mapping_TEMPLATE.csv`, `attrition_table.csv`, `household_week_draft.csv`, `demographics_clean.csv` (in `part_a_output/`) | 4.3 |
| A3 | `classify_commodities` | `commodity_mapping_CLASSIFIED.csv` (in `part_a_output/`) | 4.3, Appendix A |
| B1 | `b1_build_panel` | `hw_main.csv`, `hw_all.csv`, `trips_main.csv`, `attrition_table.csv`, `variable_dictionary.csv`, `part_b1_report.txt` | 4.3 |
| B2 | `b2_diagnostics` | `diagnostics_table.csv`, `representativeness_table.csv`, `b2_distributions.png`, `part_b2_report.txt` | 4.3, 4.4 |
| B3 | `b3_descriptives` | `b3_tables.xlsx` and one `b3_*.csv` per table, `b3_fig_weekly.png`, `b3_fig_composition.png`, `part_b3_report.txt` | 4.5 |
| B4 | `b4_models` | `b4_tables.xlsx` (sheets `Model1`, `Model2`, `Hypotheses`, `Inference_check`, `Effect_sizes`) and `b4_*.csv`, `b4_fig_stability.png`, `b4_fig_binned.png`, `part_b4_report.txt` | 4.6, 4.7 |
| B5 | `b5_robustness`, `b5_forest_figure` | `b5_tables.xlsx` (all results, one sheet per block), `b5_summary.csv`, `b5_fig_robustness.png`, `part_b5_report.txt` | 4.8 |

The key data files for extending this work are `hw_main.csv` (household-week panel, 55,180 observations on 801 households) and `trips_main.csv` (household-trip panel). They contain every variable used in the models, so later analyses can start from them without repeating the cleaning. `variable_dictionary.csv` defines every column.

## Citing

If you use this code, please cite the thesis:

> Nguro, M. D. *Mental Accounting and Slack Spending: Evidence of Discretionary Reallocation of Promotional Savings from the Dunnhumby Dataset.* BBS Financial Economics thesis, Strathmore University.

and the data source: dunnhumby, *The Complete Journey*.
