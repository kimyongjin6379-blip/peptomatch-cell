"""Data loaders for composition template and cell line databases."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import clean_numeric, project_root, load_config

logger = logging.getLogger(__name__)


def load_composition_data(
    path: str | Path | None = None,
    sheet: str = "data",
) -> pd.DataFrame:
    """Load peptone composition template xlsx."""
    if path is None:
        cfg = load_config()
        path = project_root() / cfg.get("data", {}).get(
            "composition_file", "data/composition_template.xlsx"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Composition file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet)
    logger.info(f"Loaded composition data: {df.shape[0]} samples, {df.shape[1]} columns")

    # Clean numeric columns
    skip = {"sample_id", "material_type", "Sample_name", "raw_material", "manufacturer"}
    for col in df.columns:
        if col not in skip:
            df[col] = df[col].apply(clean_numeric)

    return df


def load_cell_line_table() -> pd.DataFrame:
    """Load cell line database from config or default CSV."""
    cfg = load_config()
    path = project_root() / cfg.get("data", {}).get(
        "cell_line_file", "data/cell_lines.csv"
    )
    if path.exists():
        return pd.read_csv(path)

    # Return built-in defaults if no file
    from .cell_line_priors import CELL_LINE_DB
    rows = []
    for cl_id, info in CELL_LINE_DB.items():
        rows.append({
            "cell_line_id": cl_id,
            "name": info["name"],
            "species": info["species"],
            "organism": info["organism"],
            "kegg_org": info["kegg_org"],
            "tissue": info["tissue"],
            "application": info["application"],
        })
    return pd.DataFrame(rows)
