"""Scoring engine — match peptone supply to cell line demand.

Unlike the microorganism version which relies on genome-derived demand,
this module uses literature-based demand profiles from cell_line_priors.
The scoring emphasizes:
  1. Essential AA coverage (mandatory, no alternative)
  2. Conditionally essential AA (partial synthesis, cell-line specific)
  3. MW-based bioavailability (animal cells prefer free AA & small peptides)
  4. Vitamin/nucleotide supplementation
  5. Organic acid impact (pH, metabolic effects)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .cell_line_priors import (
    ESSENTIAL_AA, CONDITIONALLY_ESSENTIAL_AA, NON_ESSENTIAL_AA,
    get_cell_line_prior,
)
from .composition_features import CompositionFeatureExtractor, STANDARD_AA

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """Single peptone recommendation result."""
    rank: int
    sample_name: str
    total_score: float
    raw_material: str
    material_type: str
    sub_scores: dict[str, float] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


# ── Default weights (cell-line optimized) ─────────────────────────────

DEFAULT_WEIGHTS = {
    "essential_aa": 2.0,       # highest: cannot be substituted
    "conditional_aa": 1.2,     # cell-line-specific needs
    "non_essential_aa": 0.5,   # helpful but not critical
    "faa_preference": 1.5,     # animal cells prefer free AA
    "mw_low": 1.3,             # di/tripeptides → PepT transporters
    "mw_medium": 0.7,          # oligopeptides → limited uptake
    "mw_high": 0.2,            # large peptides → need extracellular protease
    "vitamin": 0.8,            # all B vitamins needed externally
    "nucleotide": 0.4,         # de novo possible, salvage preferred
    "orgacid_penalty": -0.3,   # excess organic acid → pH issues
    "hydrolysis_bonus": 0.5,   # higher FAA/TAA ratio → better for animal cells
}


class CellPeptoneRecommender:
    """Recommend peptones for animal cell lines."""

    def __init__(
        self,
        feature_extractor: CompositionFeatureExtractor,
        weights: dict[str, float] | None = None,
    ):
        self.extractor = feature_extractor
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def recommend(
        self,
        cell_line_id: str,
        sample_names: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RecommendationResult]:
        """Score all peptones for a given cell line and return top-K."""
        prior = get_cell_line_prior(cell_line_id)

        if sample_names is None:
            sample_names = self.extractor.df["Sample_name"].tolist()

        results = []
        all_scores_raw = []

        for name in sample_names:
            supply = self.extractor.get_supply(name)
            if not supply:
                continue
            score, sub = self._compute_score(supply, prior)
            all_scores_raw.append(score)

            strengths, weaknesses = self._analyze_fit(supply, prior)

            results.append(RecommendationResult(
                rank=0,
                sample_name=name,
                total_score=score,
                raw_material=supply.get("raw_material", ""),
                material_type=supply.get("material_type", ""),
                sub_scores=sub,
                strengths=strengths,
                weaknesses=weaknesses,
            ))

        # Normalize to 0-100
        if all_scores_raw:
            min_s = min(all_scores_raw)
            max_s = max(all_scores_raw)
            spread = max_s - min_s if max_s > min_s else 1
            for r in results:
                # Handle negative scores (e.g., from orgacid penalty)
                r.total_score = max(0.0, ((r.total_score - min_s) / spread) * 100)

        # Sort and assign ranks
        results.sort(key=lambda x: -x.total_score)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results[:top_k]

    def _compute_score(
        self,
        supply: dict[str, float],
        prior: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """Compute match score between peptone supply and cell line demand."""
        w = self.weights
        aa_demand = prior["aa_demand"]
        vit_demand = prior["vitamin_demand"]
        nuc_demand = prior["nucleotide_demand"]
        mw_pref = prior["mw_preference"]
        transporter = prior["transporter_profile"]

        sub_scores: dict[str, float] = {}

        # ── 1. Essential AA matching (highest weight) ──
        ess_score = 0.0
        ess_count = 0
        for aa in ESSENTIAL_AA:
            demand = aa_demand.get(aa, 1.0)
            # Prefer FAA (directly available), fallback to TAA
            faa_val = supply.get(f"supply_faa_{aa}", 0)
            taa_val = supply.get(f"supply_taa_{aa}", 0)
            # FAA is immediately available; TAA needs digestion
            aa_supply = faa_val * 1.0 + (taa_val - faa_val) * 0.3
            ess_score += aa_supply * demand
            ess_count += 1
        sub_scores["essential_aa"] = (ess_score / max(ess_count, 1)) * w["essential_aa"]

        # ── 2. Conditionally essential AA ──
        cond_score = 0.0
        cond_count = 0
        for aa in CONDITIONALLY_ESSENTIAL_AA:
            demand = aa_demand.get(aa, 0.5)
            faa_val = supply.get(f"supply_faa_{aa}", 0)
            taa_val = supply.get(f"supply_taa_{aa}", 0)
            aa_supply = faa_val * 1.0 + (taa_val - faa_val) * 0.3
            cond_score += aa_supply * demand
            cond_count += 1
        sub_scores["conditional_aa"] = (cond_score / max(cond_count, 1)) * w["conditional_aa"]

        # ── 3. Non-essential AA ──
        ne_score = 0.0
        ne_count = 0
        for aa in NON_ESSENTIAL_AA:
            demand = aa_demand.get(aa, 0.2)
            faa_val = supply.get(f"supply_faa_{aa}", 0)
            taa_val = supply.get(f"supply_taa_{aa}", 0)
            aa_supply = faa_val * 1.0 + (taa_val - faa_val) * 0.3
            ne_score += aa_supply * demand
            ne_count += 1
        sub_scores["non_essential_aa"] = (ne_score / max(ne_count, 1)) * w["non_essential_aa"]

        # ── 4. FAA preference (animal cells absorb free AA best) ──
        faa_total = supply.get("supply_faa_total", 0)
        sub_scores["faa_preference"] = faa_total * w["faa_preference"]

        # ── 5. MW distribution (weighted by cell preference) ──
        mw_low = supply.get("supply_mw_low", 0)
        mw_med = supply.get("supply_mw_medium", 0)
        mw_high = supply.get("supply_mw_high", 0)

        # Apply transporter-based preference
        free_aa_pref = transporter.get("free_aa", 0.9)
        dipep_pref = transporter.get("dipeptide", 0.5)

        mw_score = (
            mw_low * w["mw_low"] * (1 + dipep_pref)
            + mw_med * w["mw_medium"]
            + mw_high * w["mw_high"]
        )
        sub_scores["mw_bioavailability"] = mw_score

        # ── 6. Vitamin supply ──
        vit_score = 0.0
        vit_count = 0
        for vit_name, demand in vit_demand.items():
            vit_supply = supply.get(f"supply_vitamin_{vit_name}", 0)
            vit_score += vit_supply * demand
            vit_count += 1
        sub_scores["vitamin"] = (vit_score / max(vit_count, 1)) * w["vitamin"]

        # ── 7. Nucleotide supply ──
        nuc_total = supply.get("supply_nucleotide_total", 0)
        sub_scores["nucleotide"] = nuc_total * nuc_demand * w["nucleotide"]

        # ── 8. Hydrolysis degree bonus ──
        hydrolysis = supply.get("supply_hydrolysis_ratio", 0)
        sub_scores["hydrolysis_bonus"] = hydrolysis * w["hydrolysis_bonus"]

        # ── 9. Organic acid penalty ──
        orgacid = supply.get("supply_orgacid_total", 0)
        # High organic acid content can lower pH → negative impact
        orgacid_normalized = min(orgacid / 200000, 1.0)  # normalize
        sub_scores["orgacid_effect"] = orgacid_normalized * w["orgacid_penalty"]

        # ── Total ──
        total = sum(sub_scores.values())

        return total, sub_scores

    def _analyze_fit(
        self,
        supply: dict[str, float],
        prior: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Identify strengths and weaknesses of this peptone for the cell line."""
        aa_demand = prior["aa_demand"]
        strengths = []
        weaknesses = []

        # Check essential AA coverage
        for aa in ESSENTIAL_AA:
            faa = supply.get(f"supply_faa_{aa}", 0)
            taa = supply.get(f"supply_taa_{aa}", 0)
            if faa >= 1.0:
                strengths.append(f"{aa} FAA 풍부 ({faa:.1f}%)")
            elif taa < 0.5:
                weaknesses.append(f"{aa} 함량 부족 (TAA {taa:.1f}%)")

        # Check conditionally essential with high demand
        for aa in CONDITIONALLY_ESSENTIAL_AA:
            demand = aa_demand.get(aa, 0.5)
            faa = supply.get(f"supply_faa_{aa}", 0)
            if demand >= 0.7 and faa >= 0.5:
                strengths.append(f"{aa} 높은 수요 충족 (FAA {faa:.1f}%)")
            elif demand >= 0.7 and faa < 0.1:
                weaknesses.append(f"{aa} 수요 높으나 FAA 부족")

        # MW assessment
        mw_low = supply.get("supply_mw_low", 0)
        if mw_low > 0.25:
            strengths.append(f"저분자(<250Da) 비율 높음 ({mw_low:.0%})")
        mw_high = supply.get("supply_mw_high", 0)
        if mw_high > 0.4:
            weaknesses.append(f"고분자(>1000Da) 비율 높음 ({mw_high:.0%}) — 동물세포 흡수 제한")

        # Hydrolysis
        hydro = supply.get("supply_hydrolysis_ratio", 0)
        if hydro > 0.3:
            strengths.append(f"높은 가수분해도 ({hydro:.0%}) — 동물세포에 유리")

        return strengths[:5], weaknesses[:5]

    def get_score_breakdown(
        self,
        cell_line_id: str,
        sample_name: str,
    ) -> dict[str, Any]:
        """Get detailed score breakdown for a specific peptone-cell line pair."""
        prior = get_cell_line_prior(cell_line_id)
        supply = self.extractor.get_supply(sample_name)
        score, sub_scores = self._compute_score(supply, prior)

        return {
            "cell_line": cell_line_id,
            "sample_name": sample_name,
            "total_score": score,
            "sub_scores": sub_scores,
            "cell_line_info": prior["cell_line_info"],
            "special_notes": prior["special_notes"],
        }
