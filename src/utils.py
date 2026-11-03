"""Helpers shared by the pipeline steps: report logging, raw-data loading, demeaning, plot style."""
from __future__ import annotations

import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Report logging: everything a step prints is also written to a text report
# ---------------------------------------------------------------------------


class Reporter:
    """Print to the screen and keep a copy so the step can save it as ``part_*_report.txt``."""

    def __init__(self) -> None:
        self._buf = io.StringIO()

    def say(self, *args) -> None:
        text = " ".join(str(a) for a in args)
        print(text)
        self._buf.write(text + "\n")

    def header(self, title: str) -> None:
        self.say("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)

    def save(self, path: Path) -> None:
        Path(path).write_text(self._buf.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Raw data and panel files
# ---------------------------------------------------------------------------
REQUIRED_RAW = ("transaction_data", "product", "hh_demographic")


def check_raw_data(data_dir: Path) -> None:
    """Fail early, with instructions, if the Dunnhumby CSV files are not where the code expects them."""
    data_dir = Path(data_dir)
    missing = [n for n in REQUIRED_RAW if not (data_dir / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Could not find {', '.join(n + '.csv' for n in missing)} in '{data_dir.resolve()}'.\n"
            "The dataset is not distributed with this repository. Download 'The Complete Journey' "
            "(see the README, section 'Getting the data'), unzip it, and point DATA_DIR at the folder "
            "that contains transaction_data.csv, product.csv and hh_demographic.csv."
        )


def load_table(data_dir: Path, name: str) -> pd.DataFrame:
    """Read ``<data_dir>/<name>.csv`` and lower-case the column names (the raw files mix upper and lower case)."""
    df = pd.read_csv(Path(data_dir) / f"{name}.csv")
    df.columns = [c.lower() for c in df.columns]
    return df


def read_panel(path: Path) -> pd.DataFrame:
    """Read a household-week panel written by step B1, sorted by household and week."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in '{path.parent}'. Run step B1 (b1_build_panel) first.")
    return pd.read_csv(path).sort_values(["household_key", "week_no"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demographic recodes (the same coding is used in Part A and Part B)
# ---------------------------------------------------------------------------
INCOME_MIDPOINTS = {"Under 15K": 7.5, "15-24K": 19.5, "25-34K": 29.5, "35-49K": 42, "50-74K": 62, "75-99K": 87,
                    "100-124K": 112, "125-149K": 137, "150-174K": 162, "175-199K": 187, "200-249K": 224.5,
                    "250K+": 275}   # $000; the open-ended 250K+ band is set to 275 (an assumption)
HH_SIZE_MAP = {"1": 1, "2": 2, "3": 3, "4": 4, "5+": 5}   # '5+' is top-coded at 5

# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------


def demean(df: pd.DataFrame, cols: list[str], by: list[str], tol: float = 1e-10, max_iter: int = 1000) -> pd.DataFrame:
    """Within-transformation of ``cols`` absorbing the fixed effects in ``by``.

    ``by = []`` removes the grand mean, one name gives group demeaning, and two or more names use alternating
    projections, which handles the unbalanced two-way (household + week) case without dummy variables.
    """
    out = df[cols].astype(float).copy()
    if not by:
        return out - out.mean()
    if len(by) == 1:
        return out - out.groupby(df[by[0]]).transform("mean")
    for _ in range(max_iter):
        prev = out.to_numpy().copy()
        for g in by:
            out = out - out.groupby(df[g]).transform("mean")
        if np.abs(out.to_numpy() - prev).max() < tol:
            break
    return out


def star(p: float) -> str:
    """Significance stars: * p<0.10, ** p<0.05, *** p<0.01."""
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


@contextlib.contextmanager
def quiet_warnings():
    """Silence library warnings (e.g. from statsmodels) while a step runs, then restore the previous settings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------------
# Plot style (applied with ``plt.rc_context(PLOT_RC)`` so it never leaks into other figures)
# ---------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PLOT_RC = {"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
           "axes.edgecolor": GRID, "axes.titlesize": 10, "axes.titleweight": "regular", "axes.titlelocation": "left",
           "figure.facecolor": "white", "axes.facecolor": "white"}


def style_axes(ax, grid: str = "y") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def show_image(path: Path, width: int | None = None) -> None:
    """Display a saved figure inside a notebook (does nothing outside Jupyter)."""
    try:
        from IPython.display import Image, display
    except ImportError:
        return
    display(Image(filename=str(path), width=width))
