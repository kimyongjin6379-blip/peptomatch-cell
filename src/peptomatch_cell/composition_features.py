"""Extract supply features from peptone composition data.

Converts raw composition data (94 columns) into normalized supply scores
that can be matched against cell line demand profiles.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column prefix mappings
FAA_PREFIX = "faa_"
TAA_PREFIX = "taa_"
MW_PREFIX = "mw_"
VITB_PREFIX = "vitB_"
MINERAL_PREFIX = "mineral_"
NUCLEOTIDE_PREFIX = "nucleotide_"
ORGACID_PREFIX = "orgacid_"
SUGAR_PREFIX = "sugar_"


# Amino acid name mapping (column suffix → standard 3-letter)
AA_NAME_MAP = {
    "Aspartic acid": "Asp", "Threonine": "Thr", "Serine": "Ser",
    "Asparagine": "Asn", "Glutamic acid": "Glu", "Glutamine": "Gln",
    "Cysteine": "Cys", "Proline": "Pro", "Glycine": "Gly",
    "Alanine": "Ala", "Valine": "Val", "Methionine": "Met",
    "Isoleucine": "Ile", "Leucine": "Leu", "Tyrosine": "Tyr",
    "Phenylalanine": "Phe", "Histidine": "His", "Tryptophan": "Trp",
    "Lysine": "Lys", "Arginine": "Arg", "Cystine": "Cys",
    "GABA": "GABA", "Hydroxyproline": "Hyp", "Citruline": "Cit",
    "Ornithine": "Orn",
}

# Standard 20 AA for scoring
STANDARD_AA = [
    "Ala", "Arg", "Asn", "Asp", "Cys", "Glu", "Gln", "Gly",
    "His", "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser",
    "Thr", "Trp", "Tyr", "Val",
]


class CompositionFeatureExtractor:
    """Extract normalized supply features from peptone composition DataFrame."""

    def __init__(self, composition_df: pd.DataFrame):
        self.df = composition_df.copy()
        self._supply_cache: dict[str, dict[str, float]] = {}

    def get_supply(self, sample_name: str) -> dict[str, float]:
        """Get all supply features for a peptone sample."""
        if sample_name in self._supply_cache:
            return self._supply_cache[sample_name]

        row = self.df[self.df["Sample_name"] == sample_name]
        if row.empty:
            logger.warning(f"Sample not found: {sample_name}")
            return {}

        r = row.iloc[0]
        supply = {}

        # ── FAA (Free amino acids) ──
        faa_cols = [c for c in self.df.columns if c.startswith(FAA_PREFIX)]
        faa_total = sum(float(r.get(c, 0) or 0) for c in faa_cols)
        supply["supply_faa_total"] = faa_total

        # Individual FAA
        for col in faa_cols:
            aa_name_raw = col.replace(FAA_PREFIX, "")
            aa_std = AA_NAME_MAP.get(aa_name_raw, aa_name_raw)
            if aa_std in STANDARD_AA:
                supply[f"supply_faa_{aa_std}"] = float(r.get(col, 0) or 0)

        # ── TAA (Total amino acids) ──
        taa_cols = [c for c in self.df.columns if c.startswith(TAA_PREFIX)]
        taa_total = sum(float(r.get(c, 0) or 0) for c in taa_cols)
        supply["supply_taa_total"] = taa_total

        for col in taa_cols:
            aa_name_raw = col.replace(TAA_PREFIX, "")
            aa_std = AA_NAME_MAP.get(aa_name_raw, aa_name_raw)
            if aa_std in STANDARD_AA:
                supply[f"supply_taa_{aa_std}"] = float(r.get(col, 0) or 0)

        # ── FAA/TAA ratio (hydrolysis degree) ──
        supply["supply_hydrolysis_ratio"] = faa_total / taa_total if taa_total > 0 else 0

        # ── Molecular weight distribution ──
        supply["supply_mw_avg"] = float(r.get("mw_avg_Da", 0) or 0)
        supply["supply_mw_low"] = float(r.get("mw_pct_lt250Da", 0) or 0) / 100
        supply["supply_mw_medium"] = (
            float(r.get("mw_pct_250_500Da", 0) or 0)
            + float(r.get("mw_pct_500_750Da", 0) or 0)
            + float(r.get("mw_pct_750_1000Da", 0) or 0)
        ) / 100
        supply["supply_mw_high"] = float(r.get("mw_pct_gt1000Da", 0) or 0) / 100

        # ── Vitamins ──
        vit_map = {"vitB_B1": "B1", "vitB_B2": "B2", "vitB_B3": "B3",
                   "vitB_B6": "B6", "vitB_B9": "B9"}
        vit_total = 0
        for col, vname in vit_map.items():
            val = float(r.get(col, 0) or 0)
            supply[f"supply_vitamin_{vname}"] = val
            vit_total += val
        supply["supply_vitamin_total"] = vit_total

        # ── Nucleotides ──
        nuc_cols = [c for c in self.df.columns if c.startswith(NUCLEOTIDE_PREFIX)]
        nuc_total = sum(float(r.get(c, 0) or 0) for c in nuc_cols)
        supply["supply_nucleotide_total"] = nuc_total

        # ── Minerals ──
        min_cols = [c for c in self.df.columns if c.startswith(MINERAL_PREFIX)]
        for col in min_cols:
            mname = col.replace(MINERAL_PREFIX, "")
            supply[f"supply_mineral_{mname}"] = float(r.get(col, 0) or 0)

        # ── Organic acids ──
        oa_cols = [c for c in self.df.columns if c.startswith(ORGACID_PREFIX)]
        oa_total = sum(float(r.get(c, 0) or 0) for c in oa_cols)
        supply["supply_orgacid_total"] = oa_total
        for col in oa_cols:
            oa_name = col.replace(ORGACID_PREFIX, "")
            supply[f"supply_orgacid_{oa_name}"] = float(r.get(col, 0) or 0)

        # ── General ──
        supply["supply_TN"] = float(r.get("general_TN", 0) or 0)
        supply["supply_AN"] = float(r.get("general_AN", 0) or 0)
        supply["supply_AN_TN_ratio"] = (
            supply["supply_AN"] / supply["supply_TN"]
            if supply["supply_TN"] > 0 else 0
        )
        supply["supply_total_sugar"] = float(r.get("general_total_sugar", 0) or 0)
        supply["supply_reducing_sugar"] = float(r.get("general_reducing_sugar", 0) or 0)

        # ── Metadata ──
        supply["raw_material"] = str(r.get("raw_material", ""))
        supply["manufacturer"] = str(r.get("manufacturer", ""))
        supply["material_type"] = str(r.get("material_type", ""))

        self._supply_cache[sample_name] = supply
        return supply

    def get_all_supplies(self, sample_names: list[str] | None = None) -> pd.DataFrame:
        """Get supply features for multiple samples as DataFrame."""
        if sample_names is None:
            sample_names = self.df["Sample_name"].tolist()
        records = []
        for name in sample_names:
            s = self.get_supply(name)
            if s:
                s["Sample_name"] = name
                records.append(s)
        return pd.DataFrame(records)

    def normalize_supplies(self, sample_names: list[str] | None = None) -> pd.DataFrame:
        """Get min-max normalized supply features."""
        df = self.get_all_supplies(sample_names)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cmin, cmax = df[col].min(), df[col].max()
            if cmax > cmin:
                df[f"{col}_norm"] = (df[col] - cmin) / (cmax - cmin)
            else:
                df[f"{col}_norm"] = 0.5
        return df
