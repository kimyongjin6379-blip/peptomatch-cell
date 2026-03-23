"""Generate human-readable explanations for recommendations (Korean/English)."""

from __future__ import annotations

from typing import Any

from .cell_line_priors import (
    ESSENTIAL_AA, CONDITIONALLY_ESSENTIAL_AA, get_cell_line_prior,
)
from .scoring import RecommendationResult
from .blend_optimizer import BlendResult


class RecommendationExplainer:
    """Generate explanations for peptone recommendations."""

    def explain_single(
        self,
        result: RecommendationResult,
        cell_line_id: str,
        lang: str = "ko",
    ) -> str:
        """Generate explanation for a single peptone recommendation."""
        prior = get_cell_line_prior(cell_line_id)
        info = prior["cell_line_info"]

        if lang == "ko":
            lines = [
                f"### {result.sample_name} (#{result.rank}, {result.total_score:.1f}점)",
                f"**원료**: {result.raw_material} | **유형**: {result.material_type}",
                "",
            ]

            # Sub-score breakdown
            lines.append("**점수 구성:**")
            score_labels = {
                "essential_aa": "필수 아미노산 매칭",
                "conditional_aa": "조건부 필수 AA",
                "non_essential_aa": "비필수 AA",
                "faa_preference": "유리 AA 풍부도",
                "mw_bioavailability": "분자량 기반 생체이용률",
                "vitamin": "비타민 공급",
                "nucleotide": "뉴클레오타이드",
                "hydrolysis_bonus": "가수분해도 보너스",
                "orgacid_effect": "유기산 영향",
            }
            for key, label in score_labels.items():
                val = result.sub_scores.get(key, 0)
                bar = "█" * max(int(val * 10), 0) if val > 0 else "▒" * max(int(-val * 10), 1)
                lines.append(f"  - {label}: {val:.3f} {bar}")

            # Strengths
            if result.strengths:
                lines.append("\n**✅ 강점:**")
                for s in result.strengths:
                    lines.append(f"  - {s}")

            # Weaknesses
            if result.weaknesses:
                lines.append("\n**⚠️ 약점:**")
                for w in result.weaknesses:
                    lines.append(f"  - {w}")

            # Cell line notes
            notes = prior.get("special_notes", [])
            if notes:
                lines.append(f"\n**💡 {info['name']} 참고:**")
                for n in notes[:3]:
                    lines.append(f"  - {n}")

        else:  # English
            lines = [
                f"### {result.sample_name} (#{result.rank}, Score: {result.total_score:.1f})",
                f"**Material**: {result.raw_material} | **Type**: {result.material_type}",
                "",
            ]
            if result.strengths:
                lines.append("**Strengths:**")
                for s in result.strengths:
                    lines.append(f"  - {s}")
            if result.weaknesses:
                lines.append("**Weaknesses:**")
                for w in result.weaknesses:
                    lines.append(f"  - {w}")

        return "\n".join(lines)

    def explain_blend(
        self,
        result: BlendResult,
        cell_line_id: str,
        lang: str = "ko",
    ) -> str:
        """Generate explanation for a blend result."""
        prior = get_cell_line_prior(cell_line_id)

        if lang == "ko":
            lines = [
                "### 블렌딩 최적화 결과",
                "",
                "**조합:**",
            ]
            for name, ratio in zip(result.peptones, result.ratios):
                lines.append(f"  - {name}: **{ratio:.0%}**")

            lines.extend([
                "",
                f"**블렌드 점수**: {result.blend_score:.1f}",
                f"**최고 단일 점수**: {result.best_single_score:.1f} ({result.best_single_name})",
                f"**시너지**: {result.synergy:+.1f}점",
                "",
            ])

            # Synergy interpretation
            if result.synergy > 5:
                lines.append("✅ **유의미한 시너지** — 블렌딩이 효과적입니다.")
            elif result.synergy > 0:
                lines.append("🔸 **소폭 시너지** — 블렌딩이 약간 유리합니다.")
            else:
                lines.append("⚠️ **시너지 없음** — 단일 펩톤이 더 효율적일 수 있습니다.")

            # Complementarity
            comp = result.complementarity
            if comp:
                lines.extend([
                    "",
                    "**상보성 분석:**",
                    f"  - 프로파일 다양성: {comp.get('profile_diversity', 0):.2f}",
                    f"  - 특성 커버율: {comp.get('feature_coverage', 0):.0%}",
                    f"  - 필수AA 갭 충족: {comp.get('gap_filling', 0):.0%}",
                ])

            # AA coverage
            if result.aa_coverage:
                low_coverage = [
                    aa for aa, cov in result.aa_coverage.items()
                    if cov < 0.3 and aa in ESSENTIAL_AA
                ]
                if low_coverage:
                    lines.append(f"\n⚠️ **여전히 부족한 필수 AA**: {', '.join(low_coverage)}")
                else:
                    lines.append("\n✅ 모든 필수 AA가 적정 수준으로 커버됩니다.")

        else:
            lines = [
                "### Blend Optimization Result",
                "",
                f"**Blend Score**: {result.blend_score:.1f} | "
                f"**Best Single**: {result.best_single_score:.1f} | "
                f"**Synergy**: {result.synergy:+.1f}",
            ]
            for name, ratio in zip(result.peptones, result.ratios):
                lines.append(f"  - {name}: {ratio:.0%}")

        return "\n".join(lines)
