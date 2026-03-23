"""PeptoMatch Cell — Streamlit web application for animal cell line peptone matching."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np

from peptomatch_cell.utils import load_config, setup_logging
from peptomatch_cell.io_loaders import load_composition_data
from peptomatch_cell.composition_features import CompositionFeatureExtractor
from peptomatch_cell.cell_line_priors import (
    get_cell_line_prior, get_all_cell_line_ids, get_cell_line_summary,
    CELL_LINE_DB, ESSENTIAL_AA, CONDITIONALLY_ESSENTIAL_AA, NON_ESSENTIAL_AA,
)
from peptomatch_cell.scoring import CellPeptoneRecommender
from peptomatch_cell.blend_optimizer import BlendOptimizer
from peptomatch_cell.explain import RecommendationExplainer
from peptomatch_cell.cell_viz import CellViz

setup_logging()

# ── Page config ──
st.set_page_config(
    page_title="PeptoMatch Cell",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached resources ──

@st.cache_resource
def get_config():
    return load_config()

@st.cache_resource
def get_composition_df():
    return load_composition_data()

@st.cache_resource
def get_extractor():
    df = get_composition_df()
    return CompositionFeatureExtractor(df)

@st.cache_resource
def get_recommender():
    ext = get_extractor()
    cfg = get_config()
    weights = cfg.get("weights", None)
    return CellPeptoneRecommender(ext, weights)

@st.cache_resource
def get_blend_optimizer():
    return BlendOptimizer(get_extractor(), get_recommender())


def get_sempio_samples() -> list[str]:
    """Get filtered Sempio peptone sample names."""
    cfg = get_config()
    filter_list = cfg.get("peptone_filter", [])
    if filter_list:
        return [n for n in filter_list if n in get_composition_df()["Sample_name"].values]
    # Fallback: filter by manufacturer
    df = get_composition_df()
    return df[df["manufacturer"] == "Sempio"]["Sample_name"].tolist()


# ── Main ──

def main():
    st.title("🧫 PeptoMatch Cell v0.1")
    st.caption("동물 세포주 전용 펩톤 추천 시스템 — Sempio Bio")

    tabs = st.tabs([
        "🎯 펩톤 추천",
        "🧪 블렌딩 최적화",
        "🔬 세포주 프로파일",
        "📊 펩톤 탐색기",
        "⚖️ 세포주 비교",
    ])

    with tabs[0]:
        render_recommendation_tab()
    with tabs[1]:
        render_blend_tab()
    with tabs[2]:
        render_cell_line_tab()
    with tabs[3]:
        render_peptone_explorer_tab()
    with tabs[4]:
        render_comparison_tab()


# ── Tab 1: Peptone Recommendation ──

def render_recommendation_tab():
    st.header("🎯 세포주별 펩톤 추천")

    col1, col2 = st.columns([1, 2])

    with col1:
        cell_line_ids = get_all_cell_line_ids()
        selected_cl = st.selectbox(
            "세포주 선택",
            cell_line_ids,
            format_func=lambda x: f"{x} — {CELL_LINE_DB[x]['name']}",
            key="rec_cell_line",
        )

        top_k = st.slider("추천 수", 3, 16, 5, key="rec_topk")

        use_sempio_only = st.checkbox("Sempio 제품만", value=True, key="rec_sempio")

        if st.button("🔍 추천 실행", type="primary", key="rec_run"):
            samples = get_sempio_samples() if use_sempio_only else None
            recommender = get_recommender()
            results = recommender.recommend(selected_cl, sample_names=samples, top_k=top_k)

            st.session_state["rec_results"] = results
            st.session_state["rec_cell_line_id"] = selected_cl

    with col2:
        if "rec_results" in st.session_state:
            results = st.session_state["rec_results"]
            cl_id = st.session_state["rec_cell_line_id"]

            # Quick summary
            prior = get_cell_line_prior(cl_id)
            info = prior["cell_line_info"]
            st.info(f"**{info['name']}** | {info['species']} | {info['tissue']} | {info['application']}")

            # Results table
            table_data = []
            for r in results:
                table_data.append({
                    "순위": r.rank,
                    "펩톤": r.sample_name,
                    "점수": f"{r.total_score:.1f}",
                    "원료": r.raw_material,
                    "유형": r.material_type,
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            # Radar chart
            fig = CellViz.recommendation_radar(results)
            st.plotly_chart(fig, use_container_width=True, key="rec_radar")

            # Detailed explanations
            explainer = RecommendationExplainer()
            with st.expander("📋 상세 분석 보기", expanded=False):
                for r in results:
                    explanation = explainer.explain_single(r, cl_id)
                    st.markdown(explanation)
                    st.divider()


# ── Tab 2: Blend Optimization ──

def render_blend_tab():
    st.header("🧪 블렌딩 최적화")

    col1, col2 = st.columns([1, 2])

    with col1:
        cell_line_ids = get_all_cell_line_ids()
        selected_cl = st.selectbox(
            "세포주 선택",
            cell_line_ids,
            format_func=lambda x: f"{x} — {CELL_LINE_DB[x]['name']}",
            key="blend_cell_line",
        )

        mode = st.radio("모드", ["수동 선택", "자동 탐색"], key="blend_mode")

        samples = get_sempio_samples()

        if mode == "수동 선택":
            selected_peptones = st.multiselect(
                "펩톤 선택 (2~3종)",
                samples,
                max_selections=3,
                key="blend_manual",
            )

            if st.button("🔬 블렌딩 최적화", type="primary", key="blend_run"):
                if len(selected_peptones) < 2:
                    st.warning("2종 이상 선택해주세요.")
                else:
                    optimizer = get_blend_optimizer()
                    with st.spinner("최적 비율 탐색 중..."):
                        result = optimizer.optimize_blend(selected_cl, selected_peptones)
                    st.session_state["blend_result"] = result
                    st.session_state["blend_cl_id"] = selected_cl

        else:  # Auto search
            blend_size = st.radio("조합 크기", [2, 3], horizontal=True, key="blend_size")
            top_k = st.slider("상위 조합 수", 3, 10, 5, key="blend_topk")

            if st.button("🔍 자동 탐색", type="primary", key="blend_auto_run"):
                optimizer = get_blend_optimizer()
                with st.spinner(f"{len(samples)}종 중 {blend_size}종 조합 탐색 중..."):
                    results = optimizer.auto_search(
                        selected_cl, samples, blend_size=blend_size, top_k=top_k
                    )
                st.session_state["blend_auto_results"] = results
                st.session_state["blend_cl_id"] = selected_cl

    with col2:
        # Manual result
        if "blend_result" in st.session_state and mode == "수동 선택":
            result = st.session_state["blend_result"]
            _render_blend_result(result, st.session_state["blend_cl_id"])

        # Auto search results
        if "blend_auto_results" in st.session_state and mode == "자동 탐색":
            results = st.session_state["blend_auto_results"]
            cl_id = st.session_state["blend_cl_id"]

            for i, result in enumerate(results):
                with st.expander(
                    f"#{i+1} {' + '.join(result.peptones)} "
                    f"(점수: {result.blend_score:.1f}, 시너지: {result.synergy:+.1f})",
                    expanded=(i == 0),
                ):
                    _render_blend_result(result, cl_id)


def _render_blend_result(result: BlendResult, cl_id: str):
    """Render a single blend result."""
    col_a, col_b = st.columns(2)

    with col_a:
        fig = CellViz.blend_ratio_pie(result)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.metric("블렌드 점수", f"{result.blend_score:.1f}")
        st.metric("최고 단일 점수", f"{result.best_single_score:.1f} ({result.best_single_name})")
        synergy_delta = f"{result.synergy:+.1f}"
        st.metric("시너지", synergy_delta,
                  delta=synergy_delta,
                  delta_color="normal" if result.synergy > 0 else "inverse")

    fig = CellViz.blend_aa_coverage(result)
    st.plotly_chart(fig, use_container_width=True)

    explainer = RecommendationExplainer()
    st.markdown(explainer.explain_blend(result, cl_id))


# ── Tab 3: Cell Line Profile ──

def render_cell_line_tab():
    st.header("🔬 세포주 프로파일")

    selected_cl = st.selectbox(
        "세포주 선택",
        get_all_cell_line_ids(),
        format_func=lambda x: f"{x} — {CELL_LINE_DB[x]['name']}",
        key="profile_cell_line",
    )

    prior = get_cell_line_prior(selected_cl)
    info = prior["cell_line_info"]

    # Info cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("종(Species)", info["organism"])
    col2.metric("조직(Tissue)", info["tissue"])
    col3.metric("KEGG Code", info["kegg_org"])
    col4.metric("뉴클레오타이드 수요", f"{prior['nucleotide_demand']:.0%}")

    st.markdown(f"**용도:** {info['application']}")

    # Special notes
    notes = prior.get("special_notes", [])
    if notes:
        st.subheader("💡 세포주 특이사항")
        for n in notes:
            st.markdown(f"- {n}")

    # Demand heatmap
    st.subheader("영양소 요구도 Overview")
    fig = CellViz.demand_heatmap(selected_cl)
    st.plotly_chart(fig, use_container_width=True, key="profile_heatmap")

    # AA demand bar chart
    st.subheader("아미노산 요구도 상세")
    fig = CellViz.aa_demand_bar(selected_cl)
    st.plotly_chart(fig, use_container_width=True, key="profile_aa_bar")

    # Transporter profile
    st.subheader("펩타이드 수송체 프로파일")
    trans = prior["transporter_profile"]
    trans_df = pd.DataFrame([
        {"수송 형태": k, "흡수 효율": f"{v:.0%}", "값": v}
        for k, v in trans.items()
    ])
    st.dataframe(trans_df[["수송 형태", "흡수 효율"]], use_container_width=True, hide_index=True)

    # MW preference
    st.subheader("분자량 선호도 가중치")
    mw = prior["mw_preference"]
    mw_df = pd.DataFrame([
        {"분자량 구간": f"{k} MW", "가중치": v}
        for k, v in mw.items()
    ])
    st.dataframe(mw_df, use_container_width=True, hide_index=True)


# ── Tab 4: Peptone Explorer ──

def render_peptone_explorer_tab():
    st.header("📊 펩톤 탐색기")

    samples = get_sempio_samples()
    selected = st.multiselect("펩톤 선택 (비교)", samples, default=samples[:3], key="explore_select")

    if not selected:
        st.info("비교할 펩톤을 선택해주세요.")
        return

    extractor = get_extractor()

    # MW comparison
    st.subheader("분자량 분포 비교")
    fig = CellViz.mw_comparison(selected, extractor)
    st.plotly_chart(fig, use_container_width=True, key="explore_mw")

    # FAA comparison
    st.subheader("유리 아미노산 (FAA) 비교")
    aa_list = ESSENTIAL_AA + CONDITIONALLY_ESSENTIAL_AA + NON_ESSENTIAL_AA
    fig_faa = go.Figure()
    for name in selected:
        supply = extractor.get_supply(name)
        vals = [supply.get(f"supply_faa_{aa}", 0) for aa in aa_list]
        fig_faa.add_trace(go.Bar(name=name, x=aa_list, y=vals))
    fig_faa.update_layout(barmode="group", yaxis_title="FAA (%)", height=450)
    st.plotly_chart(fig_faa, use_container_width=True, key="explore_faa")

    # General characteristics table
    st.subheader("일반 특성 비교")
    general_data = []
    for name in selected:
        supply = extractor.get_supply(name)
        general_data.append({
            "펩톤": name,
            "원료": supply.get("raw_material", ""),
            "TN (%)": f"{supply.get('supply_TN', 0):.2f}",
            "AN (%)": f"{supply.get('supply_AN', 0):.2f}",
            "AN/TN": f"{supply.get('supply_AN_TN_ratio', 0):.2f}",
            "FAA 총량": f"{supply.get('supply_faa_total', 0):.1f}",
            "TAA 총량": f"{supply.get('supply_taa_total', 0):.1f}",
            "가수분해도": f"{supply.get('supply_hydrolysis_ratio', 0):.0%}",
            "평균 MW (Da)": f"{supply.get('supply_mw_avg', 0):.0f}",
        })
    st.dataframe(pd.DataFrame(general_data), use_container_width=True, hide_index=True)


# ── Tab 5: Cell Line Comparison ──

def render_comparison_tab():
    st.header("⚖️ 세포주 비교")

    cell_line_ids = get_all_cell_line_ids()
    selected_cls = st.multiselect(
        "비교할 세포주 선택",
        cell_line_ids,
        default=cell_line_ids[:3],
        format_func=lambda x: f"{x} — {CELL_LINE_DB[x]['name']}",
        key="compare_cls",
    )

    if len(selected_cls) < 2:
        st.info("2개 이상 세포주를 선택해주세요.")
        return

    # Comparison chart
    fig = CellViz.compare_cell_lines(selected_cls)
    st.plotly_chart(fig, use_container_width=True, key="compare_chart")

    # Difference highlight
    st.subheader("세포주간 차이점")
    all_aa = ESSENTIAL_AA + CONDITIONALLY_ESSENTIAL_AA + NON_ESSENTIAL_AA
    diff_data = []
    for aa in all_aa:
        demands = {}
        for cl in selected_cls:
            prior = get_cell_line_prior(cl)
            demands[cl] = prior["aa_demand"].get(aa, 0)

        vals = list(demands.values())
        if max(vals) - min(vals) >= 0.2:  # significant difference
            row = {"AA": aa, "차이": f"{max(vals) - min(vals):.0%}"}
            for cl in selected_cls:
                row[cl] = f"{demands[cl]:.0%}"
            diff_data.append(row)

    if diff_data:
        st.dataframe(pd.DataFrame(diff_data), use_container_width=True, hide_index=True)
        st.caption("※ 세포주간 요구도 차이가 20%p 이상인 아미노산만 표시")
    else:
        st.success("선택한 세포주들의 AA 요구도 차이가 크지 않습니다.")

    # Application comparison
    st.subheader("세포주 용도 비교")
    app_data = []
    for cl in selected_cls:
        info = CELL_LINE_DB[cl]
        app_data.append({
            "세포주": cl,
            "유래": info["organism"],
            "조직": info["tissue"],
            "용도": info["application"],
        })
    st.dataframe(pd.DataFrame(app_data), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    import plotly.graph_objects as go
    main()
