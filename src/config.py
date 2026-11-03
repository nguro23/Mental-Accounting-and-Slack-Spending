"""Folder layout used by every step of the pipeline.

Only ``data_dir`` (the raw Dunnhumby CSV files) ever needs changing. The two output folders keep the
names used in the thesis (Appendix B): ``part_a_output`` and ``part_b_output``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    data_dir: Path   # raw Dunnhumby CSVs (not redistributed; see README)
    out_a: Path      # Part A: data checks and commodity classification
    out_b: Path      # Part B: analysis panels, diagnostics, models, robustness
    root: Path = REPO_ROOT

    @classmethod
    def default(cls, data_dir: str | os.PathLike | None = None, root: str | os.PathLike = REPO_ROOT) -> "Paths":
        """Resolve all folders relative to the repository root.

        The raw-data folder is taken from, in order: the ``data_dir`` argument, the environment variable
        ``DUNNHUMBY_DATA_DIR``, or ``<repo>/data/raw``.
        """
        root = Path(root).resolve()
        raw = data_dir or os.environ.get("DUNNHUMBY_DATA_DIR") or root / "data" / "raw"
        return cls(data_dir=Path(raw).expanduser(), out_a=root / "part_a_output", out_b=root / "part_b_output",
                   root=root)

    def make_dirs(self) -> None:
        self.out_a.mkdir(parents=True, exist_ok=True)
        self.out_b.mkdir(parents=True, exist_ok=True)

    def show(self, path: str | os.PathLike) -> str:
        """Path as printed in the reports: relative to the repository root, so no personal folders leak into them."""
        p = Path(path)
        try:
            return p.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(p)

    # ---- files passed from one step to the next -------------------------------------------------
    @property
    def template(self) -> Path:
        return self.out_a / "commodity_mapping_TEMPLATE.csv"

    @property
    def classified(self) -> Path:
        return self.out_a / "commodity_mapping_CLASSIFIED.csv"

    @property
    def household_week_draft(self) -> Path:
        return self.out_a / "household_week_draft.csv"

    @property
    def hw_all(self) -> Path:
        return self.out_b / "hw_all.csv"

    @property
    def hw_main(self) -> Path:
        return self.out_b / "hw_main.csv"

    @property
    def trips_main(self) -> Path:
        return self.out_b / "trips_main.csv"
