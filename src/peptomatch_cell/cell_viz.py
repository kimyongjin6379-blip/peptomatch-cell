"""Visualization module for cell line demand profiles and peptone matching."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .cell_line_priors import (
    ESSENTIAL_AA, CONDITIONALLY_ESSENTIAL_AA, NON_ESSENTIAL_AA,
    STANDARD_AA, get_cell_line_prior, get_all_cell_line_ids,
)
from .scoring import RecommendationResult
from .blend_optimizer import BlendResult


class CellViz:
    """Plotly visualizations for PeptoMatch Cell."""

    # ── Demand profile overview ──

    @staticmethod
    def demand_heatmap(cell_line_id: str) -> go.Figure:
        """Single-row heatmap showing all demand values for a cell line."""
        prior = get_cell_line_prior(cell_line_id)
        aa_demand = prior["aa_demand"]
        vit_demand = prior["vitamin_demand"]

        categories = []
        values = []

        for aa in ESSENTIAL_AA:
            categories.append(aa)
            values.append(aa_demand.get(aa, 1.0))

        categories.append(" ")
        values.append(float("nan"))

        for aa in CONDITIONALLY_ESSENTIAL_AA:
            categories.append(aa)
            values.append(aa_demand.get(aa, 0.5))

        categories.append("  ")
        values.append(float("nan"))

        for aa in NON_ESSENTIAL_AA:
            categories.append(aa)
            values.append(aa_demand.get(aa, 0.2))

        categories.append("   ")
        values.append(float("nan"))

        for vit in sorted(vit_demand.keys()):
            categories.append(vit)
            values.append(vit_demand[vit])

        categories.append("    ")
        values.append(float("nan"))
        categories.append("Nucleotide")
        values.append(prior["nucleotide_demand"])

        label = prior["cell_line_info"]["name"]
        text_row = [f"{v:.0%}" if v == v and v is not None else "" for v in values]

        fig = go.Figure(data=go.Heatmap(
            z=[values], x=categories, y=[label],
            colorscale="RdYlGn_r",  # reversed: red=high demand
            zmin=0, zmax=1,
            colorbar_title="Demand",
            text=[text_row], texttemplate="%{text}",
            hovertemplate="<b>%{x}</b>: %{text}<extra></extra>",
            xgap=1, ygap=1,
        ))

        n_ess = len(ESSENTIAL_AA)
        n_cond = len(CONDITIONALLY_ESSENTIAL_AA)
        n_ne = len(NON_ESSENTIAL_AA)

        fig.update_layout(
            title=f"영양소 요구도 — {label}",
            height=260,
            xaxis=dict(tickangle=-45, side="bottom"),
            annotations=[
                dict(x=n_ess / 2 - 0.5, y=1.18, text="<b>Essential</b>",
                     showarrow=False, xref="x", yref="paper", font=dict(size=10, color="red")),
                dict(x=n_ess + 1 + n_cond / 2 - 0.5, y=1.18, text="<b>Conditional</b>",
                     showarrow=False, xref="x", yref="paper", font=dict(size=10, color="orange")),
                dict(x=n_ess + 1 + n_cond + 1 + n_ne / 2 - 0.5, y=1.18, text="<b>Non-essential</b>",
                     showarrow=False, xref="x", yref="paper", font=dict(size=10, color="green")),
            ],
            margin=dict(t=80, b=10),
        )
        return fig

    # ── AA demand bar chart ──

    @staticmethod
    def aa_demand_bar(cell_line_id: str) -> go.Figure:
        """Bar chart of AA demand colored by category."""
        prior = get_cell_line_prior(cell_line_id)
        aa_demand = prior["aa_demand"]

        aa_list = ESSENTIAL_AA + CONDITIONALLY_ESSENTIAL_AA + NON_ESSENTIAL_AA
        values = [aa_demand.get(aa, 0) for aa in aa_list]
        colors = (
            ["#D32F2F"] * len(ESSENTIAL_AA)
            + ["#FF9800"] * len(CONDITIONALLY_ESSENTIAL_AA)
            + ["#4CAF50"] * len(NON_ESSENTIAL_AA)
        )

        fig = go.Figure(data=go.Bar(
            x=aa_list, y=values,
            marker_color=colors,
            text=[f"{v:.0%}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"아미노산 요구도 — {prior['cell_line_info']['name']}",
            yaxis=dict(title="Demand (0=자체합성, 1=외부공급 필수)", range=[0, 1.15]),
            height=400,
        )
        return fig

    # ── Multi cell line comparison ──

    @staticmethod
    def compare_cell_lines(cell_line_ids: list[str]) -> go.Figure:
        """Grouped bar chart comparing demand across cell lines."""
        fig = go.Figure()

        for cl_id in cell_line_ids:
            prior = get_cell_line_prior(cl_id)
            aa_demand = prior["aa_demand"]
            aa_list = ESSENTIAL_AA + CONDITIONALLY_ESSENTIAL_AA + NON_ESSENTIAL_AA
            values = [aa_demand.get(aa, 0) for aa in aa_list]

            fig.add_trace(go.Bar(
                name=cl_id,
                x=aa_list,
                y=values,
                text=[f"{v:.0%}" for v in values],
                textposition="outside",
            ))

        fig.update_layout(
            title="세포주간 AA 요구도 비교",
            barmode="group",
            yaxis=dict(title="Demand", range=[0, 1.15]),
            height=500,
        )
        return fig

    # ── Recommendation radar chart ──

    @staticmethod
    def recommendation_radar(results: list[RecommendationResult]) -> go.Figure:
        """Radar chart comparing top recommendations by sub-scores."""
        categories = [
            "필수AA", "조건부AA", "비필수AA",
            "유리AA", "분자량", "비타민", "뉴클레오타이드",
        ]
        score_keys = [
            "essential_aa", "conditional_aa", "non_essential_aa",
            "faa_preference", "mw_bioavailability", "vitamin", "nucleotide",
        ]

        fig = go.Figure()
        for res in results[:5]:
            vals = [max(res.sub_scores.get(k, 0), 0) for k in score_keys]
            # Normalize to 0-1 for radar
            max_v = max(vals) if max(vals) > 0 else 1
            vals_norm = [v / max_v for v in vals]
            vals_norm.append(vals_norm[0])  # close the polygon

            fig.add_trace(go.Scatterpolar(
                r=vals_norm,
                theta=categories + [categories[0]],
                name=f"#{res.rank} {res.sample_name}",
                fill="toself",
                opacity=0.6,
            ))

        fig.update_layout(
            title="펩톤 추천 비교 (레이더)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500,
        )
        return fig

    # ── Blend visualization ──

    @staticmethod
    def blend_ratio_pie(result: BlendResult) -> go.Figure:
        """Pie chart showing blend ratios."""
        fig = go.Figure(data=go.Pie(
            labels=result.peptones,
            values=[r * 100 for r in result.ratios],
            textinfo="label+percent",
            marker=dict(colors=["#D32F2F", "#1976D2", "#388E3C"][:len(result.peptones)]),
        ))
        fig.update_layout(
            title=f"최적 블렌딩 비율 (점수: {result.blend_score:.1f})",
            height=350,
        )
        return fig

    @staticmethod
    def blend_aa_coverage(result: BlendResult) -> go.Figure:
        """Bar chart showing AA coverage of the blend."""
        if not result.aa_coverage:
            return go.Figure()

        aa_list = ESSENTIAL_AA + CONDITIONALLY_ESSENTIAL_AA + NON_ESSENTIAL_AA
        coverages = [result.aa_coverage.get(aa, 0) for aa in aa_list]
        colors = []
        for aa, cov in zip(aa_list, coverages):
            if aa in ESSENTIAL_AA:
                colors.append("#D32F2F" if cov < 0.3 else "#4CAF50")
            else:
                colors.append("#FF9800" if cov < 0.3 else "#81C784")

        fig = go.Figure(data=go.Bar(
            x=aa_list, y=coverages,
            marker_color=colors,
            text=[f"{c:.0%}" for c in coverages],
            textposition="outside",
        ))
        fig.update_layout(
            title="블렌드 AA 커버리지",
            yaxis=dict(title="Coverage", range=[0, 1.15]),
            height=400,
        )
        fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                      annotation_text="최소 권장선")
        return fig

    # ── MW distribution comparison ──

    @staticmethod
    def mw_comparison(
        sample_names: list[str],
        extractor,
    ) -> go.Figure:
        """Stacked bar chart comparing MW distributions."""
        fig = go.Figure()
        mw_labels = ["<250Da", "250-1000Da", ">1000Da"]
        mw_colors = ["#4CAF50", "#FF9800", "#D32F2F"]

        for name in sample_names:
            supply = extractor.get_supply(name)
            if not supply:
                continue
            mw_vals = [
                supply.get("supply_mw_low", 0) * 100,
                supply.get("supply_mw_medium", 0) * 100,
                supply.get("supply_mw_high", 0) * 100,
            ]
            for i, (label, color) in enumerate(zip(mw_labels, mw_colors)):
                fig.add_trace(go.Bar(
                    name=label if name == sample_names[0] else None,
                    x=[name], y=[mw_vals[i]],
                    marker_color=color,
                    showlegend=(name == sample_names[0]),
                    legendgroup=label,
                ))

        fig.update_layout(
            title="분자량 분포 비교 (동물세포: 저분자 선호)",
            barmode="stack",
            yaxis=dict(title="%"),
            height=400,
        )
        return fig
