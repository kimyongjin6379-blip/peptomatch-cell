"""Blend optimizer for animal cell lines — find optimal 2-3 peptone combinations.

Animal cell-specific considerations:
- Free AA preference is stronger than microorganisms
- MW distribution matters more (limited peptide transporter expression)
- Organic acid accumulation can be problematic (pH sensitivity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .cell_line_priors import get_cell_line_prior, ESSENTIAL_AA
from .composition_features import STANDARD_AA
from .composition_features import CompositionFeatureExtractor
from .scoring import CellPeptoneRecommender

logger = logging.getLogger(__name__)


@dataclass
class BlendResult:
    """Result of blend optimization."""
    peptones: list[str]
    ratios: list[float]
    blend_score: float
    best_single_score: float
    best_single_name: str
    synergy: float
    sub_scores: dict[str, float] = field(default_factory=dict)
    complementarity: dict[str, float] = field(default_factory=dict)
    aa_coverage: dict[str, float] = field(default_factory=dict)


class BlendOptimizer:
    """Optimize peptone blend ratios for animal cell lines."""

    def __init__(
        self,
        feature_extractor: CompositionFeatureExtractor,
        recommender: CellPeptoneRecommender,
    ):
        self.extractor = feature_extractor
        self.recommender = recommender

    def optimize_blend(
        self,
        cell_line_id: str,
        peptone_names: list[str],
        n_restarts: int = 5,
    ) -> BlendResult:
        """Find optimal blend ratios for given peptones."""
        prior = get_cell_line_prior(cell_line_id)
        n = len(peptone_names)

        if n < 2 or n > 3:
            raise ValueError("Blend requires 2-3 peptones")

        # Get supply profiles
        supplies = []
        for name in peptone_names:
            s = self.extractor.get_supply(name)
            if not s:
                raise ValueError(f"Peptone not found: {name}")
            supplies.append(s)

        # Get single best score for synergy calculation
        single_results = self.recommender.recommend(
            cell_line_id, sample_names=peptone_names, top_k=n
        )
        best_single = max(single_results, key=lambda r: r.total_score)

        # Optimize ratios
        best_score = -np.inf
        best_ratios = [1.0 / n] * n

        def objective(ratios_raw):
            # Softmax to ensure sum=1 and all positive
            ratios = np.exp(ratios_raw) / np.sum(np.exp(ratios_raw))
            blended = self._blend_supplies(supplies, ratios.tolist())
            score, _ = self.recommender._compute_score(blended, prior)
            return -score  # minimize negative

        for _ in range(n_restarts):
            x0 = np.random.randn(n) * 0.5
            try:
                res = minimize(
                    objective,
                    x0,
                    method="Nelder-Mead",
                    options={"maxiter": 500},
                )
                if -res.fun > best_score:
                    best_score = -res.fun
                    ratios = np.exp(res.x) / np.sum(np.exp(res.x))
                    best_ratios = ratios.tolist()
            except Exception as e:
                logger.warning(f"Optimization restart failed: {e}")
                continue

        # Ensure minimum ratio (practical blending)
        min_ratio = 0.1
        best_ratios = [max(r, min_ratio) for r in best_ratios]
        total = sum(best_ratios)
        best_ratios = [r / total for r in best_ratios]

        # Final evaluation
        blended_supply = self._blend_supplies(supplies, best_ratios)
        final_score, sub_scores = self.recommender._compute_score(blended_supply, prior)

        # Normalize score similar to single recommendation
        raw_singles = []
        for name in peptone_names:
            s = self.extractor.get_supply(name)
            sc, _ = self.recommender._compute_score(s, prior)
            raw_singles.append(sc)
        max_raw = max(max(raw_singles), final_score) if raw_singles else final_score
        norm_factor = 100 / max_raw if max_raw > 0 else 1

        normalized_blend = final_score * norm_factor
        normalized_best_single = best_single.total_score

        # AA coverage analysis
        aa_coverage = self._analyze_aa_coverage(blended_supply, prior)

        # Complementarity analysis
        complementarity = self._analyze_complementarity(supplies, peptone_names)

        return BlendResult(
            peptones=peptone_names,
            ratios=best_ratios,
            blend_score=normalized_blend,
            best_single_score=normalized_best_single,
            best_single_name=best_single.sample_name,
            synergy=normalized_blend - normalized_best_single,
            sub_scores=sub_scores,
            complementarity=complementarity,
            aa_coverage=aa_coverage,
        )

    def auto_search(
        self,
        cell_line_id: str,
        candidate_names: list[str],
        blend_size: int = 2,
        top_k: int = 5,
    ) -> list[BlendResult]:
        """Search all combinations and return top-K blends."""
        if blend_size < 2 or blend_size > 3:
            raise ValueError("blend_size must be 2 or 3")

        all_results = []
        combos = list(combinations(candidate_names, blend_size))
        logger.info(f"Searching {len(combos)} combinations of {blend_size} peptones")

        for combo in combos:
            try:
                result = self.optimize_blend(
                    cell_line_id,
                    list(combo),
                    n_restarts=3,
                )
                all_results.append(result)
            except Exception as e:
                logger.warning(f"Failed for {combo}: {e}")
                continue

        all_results.sort(key=lambda r: -r.blend_score)
        return all_results[:top_k]

    def _blend_supplies(
        self,
        supplies: list[dict[str, float]],
        ratios: list[float],
    ) -> dict[str, float]:
        """Create weighted-average supply from multiple peptones."""
        blended: dict[str, float] = {}
        numeric_keys = set()
        for s in supplies:
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    numeric_keys.add(k)

        for key in numeric_keys:
            blended[key] = sum(
                ratios[i] * supplies[i].get(key, 0)
                for i in range(len(supplies))
            )

        return blended

    def _analyze_aa_coverage(
        self,
        supply: dict[str, float],
        prior: dict[str, Any],
    ) -> dict[str, float]:
        """Analyze how well the blend covers each AA demand."""
        aa_demand = prior["aa_demand"]
        coverage = {}
        for aa in STANDARD_AA:
            demand = aa_demand.get(aa, 0.5)
            faa = supply.get(f"supply_faa_{aa}", 0)
            taa = supply.get(f"supply_taa_{aa}", 0)
            effective = faa + (taa - faa) * 0.3
            # Coverage ratio (capped at 1.0)
            coverage[aa] = min(effective / max(demand * 2, 0.1), 1.0)
        return coverage

    def _analyze_complementarity(
        self,
        supplies: list[dict[str, float]],
        names: list[str],
    ) -> dict[str, float]:
        """Measure how complementary the peptones are."""
        n = len(supplies)

        # Profile diversity (Euclidean distance of AA profiles)
        aa_profiles = []
        for s in supplies:
            profile = [s.get(f"supply_faa_{aa}", 0) for aa in STANDARD_AA]
            aa_profiles.append(np.array(profile))

        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(aa_profiles[i] - aa_profiles[j])
                distances.append(d)
        profile_diversity = np.mean(distances) if distances else 0

        # Feature coverage (how many features are non-zero in at least one)
        all_keys = set()
        for s in supplies:
            all_keys.update(k for k, v in s.items()
                           if isinstance(v, (int, float)) and v > 0)
        total_keys = set()
        for s in supplies:
            total_keys.update(k for k in s if isinstance(s.get(k), (int, float)))
        coverage = len(all_keys) / max(len(total_keys), 1)

        # Essential AA gap filling
        gap_fills = 0
        total_gaps = 0
        for aa in ESSENTIAL_AA:
            key = f"supply_faa_{aa}"
            vals = [s.get(key, 0) for s in supplies]
            if min(vals) < 0.1:  # one peptone has a gap
                total_gaps += 1
                if max(vals) >= 0.3:  # another fills it
                    gap_fills += 1

        gap_filling = gap_fills / max(total_gaps, 1)

        return {
            "profile_diversity": float(profile_diversity),
            "feature_coverage": float(coverage),
            "gap_filling": float(gap_filling),
            "overall": (
                profile_diversity * 0.3
                + coverage * 0.4
                + gap_filling * 0.3
            ),
        }
