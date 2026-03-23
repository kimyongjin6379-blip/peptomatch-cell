"""Animal cell line metabolic priors — literature-based demand profiles.

Unlike microorganisms where genome completeness drives demand,
animal cells have well-characterized essential/non-essential AA requirements.
The genome contains most biosynthetic genes, but many are evolutionarily
silenced or tissue-specific. Demand is therefore set by literature consensus
and cell-line-specific characteristics.

References:
- Eagle, H. (1959) Science 130:432 — Essential AA for mammalian cells
- Ritacco et al. (2018) Biotechnol Bioeng — CHO cell metabolism
- Altamirano et al. (2006) J Biotechnol — CHO nutrient requirements
- Clincke et al. (2013) BMC Proc — Fed-batch CHO optimization
"""

from __future__ import annotations

from typing import Any


# ── Essential amino acid classification (all mammalian cells) ─────────

ESSENTIAL_AA = ["His", "Ile", "Leu", "Lys", "Met", "Phe", "Thr", "Trp", "Val"]
CONDITIONALLY_ESSENTIAL_AA = ["Arg", "Cys", "Gln", "Pro", "Tyr"]
NON_ESSENTIAL_AA = ["Ala", "Asp", "Asn", "Glu", "Gly", "Ser"]

ALL_AA = ESSENTIAL_AA + CONDITIONALLY_ESSENTIAL_AA + NON_ESSENTIAL_AA


# ── Base demand (common to all mammalian cells) ──────────────────────

_BASE_MAMMALIAN = {
    "aa_demand": {
        # Essential — cannot synthesize, demand = 1.0
        "His": 1.0, "Ile": 1.0, "Leu": 1.0, "Lys": 1.0,
        "Met": 1.0, "Phe": 1.0, "Thr": 1.0, "Trp": 1.0, "Val": 1.0,
        # Conditionally essential — partial synthesis, varies by cell line
        "Arg": 0.6, "Cys": 0.7, "Gln": 0.8, "Pro": 0.5, "Tyr": 0.5,
        # Non-essential — can synthesize but external supply helps
        "Ala": 0.2, "Asp": 0.2, "Asn": 0.3, "Glu": 0.3, "Gly": 0.3, "Ser": 0.3,
    },
    "vitamin_demand": {
        # Animal cells cannot synthesize B vitamins
        "B1": 1.0, "B2": 1.0, "B3": 1.0, "B5": 1.0,
        "B6": 1.0, "B7": 1.0, "B9": 1.0, "B12": 1.0,
        # Vitamin C: most animal cells lost ability (except some)
        "C": 0.8,
    },
    "nucleotide_demand": 0.4,  # de novo synthesis works, but salvage pathway preferred
    "transporter_profile": {
        "free_aa": 0.9,           # excellent free AA uptake
        "dipeptide": 0.5,         # PepT1/PepT2 variable expression
        "oligopeptide": 0.3,      # limited oligopeptide uptake
        "protein_fragment": 0.1,  # requires extracellular protease
    },
}

_BASE_AVIAN = {
    "aa_demand": {
        # Avian cells: same essential AA + Glycine is essential in birds
        "His": 1.0, "Ile": 1.0, "Leu": 1.0, "Lys": 1.0,
        "Met": 1.0, "Phe": 1.0, "Thr": 1.0, "Trp": 1.0, "Val": 1.0,
        "Arg": 0.8, "Cys": 0.7, "Gln": 0.7, "Pro": 0.6, "Tyr": 0.5,
        "Ala": 0.2, "Asp": 0.2, "Asn": 0.3, "Glu": 0.3,
        "Gly": 0.8,  # birds cannot synthesize enough Gly
        "Ser": 0.3,
    },
    "vitamin_demand": {
        "B1": 1.0, "B2": 1.0, "B3": 1.0, "B5": 1.0,
        "B6": 1.0, "B7": 1.0, "B9": 1.0, "B12": 1.0,
        "C": 0.3,  # chickens can synthesize vitamin C
    },
    "nucleotide_demand": 0.5,  # uric acid metabolism in birds
    "transporter_profile": {
        "free_aa": 0.9,
        "dipeptide": 0.5,
        "oligopeptide": 0.3,
        "protein_fragment": 0.1,
    },
}


# ── Cell line database with specific adjustments ─────────────────────

CELL_LINE_DB: dict[str, dict[str, Any]] = {
    "CHO": {
        "name": "CHO (Chinese Hamster Ovary)",
        "species": "Cricetulus griseus",
        "organism": "Chinese hamster",
        "kegg_org": "cge",
        "tissue": "Ovary",
        "application": "재조합 단백질, 항체 (mAb) 생산",
        "base": "mammalian",
        "aa_adjustments": {
            # CHO uses Gln as primary energy source (glutaminolysis)
            "Gln": 0.95,  # very high demand — energy + nitrogen source
            # Cys is critical for disulfide bonds in mAb production
            "Cys": 0.85,
            # Asn important for N-glycosylation
            "Asn": 0.6,
            # Pro synthesis is weak in CHO
            "Pro": 0.7,
            # Ser consumed for one-carbon metabolism
            "Ser": 0.5,
        },
        "special_notes": [
            "Gln은 에너지원(glutaminolysis)으로 대량 소모",
            "항체 생산 시 Cys(이황화결합), Asn(N-glycosylation) 추가 수요",
            "Lactate/ammonia 축적 → pH 관리 중요",
            "저분자 펩타이드 흡수 능력 중간",
        ],
        "mw_preference": {"low": 1.2, "medium": 0.8, "high": 0.4},
    },

    "Hybridoma": {
        "name": "Hybridoma (Mouse Myeloma × B-cell)",
        "species": "Mus musculus (hybrid)",
        "organism": "Mouse hybrid",
        "kegg_org": "mmu",
        "tissue": "Immune (B-cell fusion)",
        "application": "단클론항체 (mAb) 생산",
        "base": "mammalian",
        "aa_adjustments": {
            # IgG heavy production → specific AA consumption
            "Gln": 0.95,  # primary energy source
            "Ser": 0.7,   # IgG variable region enriched
            "Thr": 1.0,   # IgG constant region
            "Cys": 0.9,   # disulfide bonds (inter/intra chain)
            "Asp": 0.5,   # CDR region
            "Asn": 0.5,   # N-glycosylation at Fc region
        },
        "special_notes": [
            "IgG 생산량에 비례하여 특정 AA 소모 급증",
            "Gln 소모 매우 높음 (에너지 + 질소원)",
            "Serum-free 배양 시 펩톤 의존도 증가",
        ],
        "mw_preference": {"low": 1.0, "medium": 0.8, "high": 0.5},
    },

    "BHK-21": {
        "name": "BHK-21 (Baby Hamster Kidney)",
        "species": "Mesocricetus auratus",
        "organism": "Syrian golden hamster",
        "kegg_org": "mau",
        "tissue": "Kidney",
        "application": "바이러스 백신 생산, 재조합 단백질",
        "base": "mammalian",
        "aa_adjustments": {
            "Gln": 0.85,
            "Cys": 0.7,
            "Ser": 0.4,
            "Pro": 0.5,
        },
        "special_notes": [
            "부착 및 부유 배양 모두 가능",
            "바이러스 증식 시 뉴클레오타이드 수요 증가",
            "비교적 영양 요구도가 낮은 편",
        ],
        "nucleotide_adjustment": 0.6,  # virus production needs more
        "mw_preference": {"low": 1.0, "medium": 0.9, "high": 0.5},
    },

    "MDCK": {
        "name": "MDCK (Madin-Darby Canine Kidney)",
        "species": "Canis lupus familiaris",
        "organism": "Dog",
        "kegg_org": "cfa",
        "tissue": "Kidney (epithelial)",
        "application": "인플루엔자 백신 생산",
        "base": "mammalian",
        "aa_adjustments": {
            "Gln": 0.85,
            "Cys": 0.65,
            "Arg": 0.7,   # epithelial cells use more Arg
            "Ser": 0.45,
        },
        "special_notes": [
            "인플루엔자 바이러스 증식 숙주세포",
            "바이러스 감염 시 AA 소모 패턴 변화",
            "부착 배양 (microcarrier) 주로 사용",
            "Trypsin 처리 필요 → 외부 protease 영향",
        ],
        "nucleotide_adjustment": 0.7,  # influenza replication
        "mw_preference": {"low": 1.0, "medium": 0.8, "high": 0.5},
    },

    "PK-15": {
        "name": "PK-15 (Porcine Kidney)",
        "species": "Sus scrofa",
        "organism": "Pig",
        "kegg_org": "ssc",
        "tissue": "Kidney",
        "application": "돼지 바이러스 백신 생산 (PCV, PRRS 등)",
        "base": "mammalian",
        "aa_adjustments": {
            "Gln": 0.8,
            "Cys": 0.65,
            "Arg": 0.65,
            "Pro": 0.5,
        },
        "special_notes": [
            "돼지 유래 바이러스(PCV2, PRRS) 증식",
            "부착 배양 기반",
            "비교적 강건한 세포주",
        ],
        "nucleotide_adjustment": 0.6,
        "mw_preference": {"low": 1.0, "medium": 0.9, "high": 0.5},
    },

    "VERO": {
        "name": "VERO (African Green Monkey Kidney)",
        "species": "Chlorocebus sabaeus",
        "organism": "African green monkey",
        "kegg_org": "csab",
        "tissue": "Kidney (epithelial)",
        "application": "광범위 바이러스 백신 (폴리오, 광견병, COVID 등)",
        "base": "mammalian",
        "aa_adjustments": {
            "Gln": 0.85,
            "Cys": 0.7,
            "Arg": 0.7,
            "Asn": 0.45,
            "Ser": 0.45,
        },
        "special_notes": [
            "가장 널리 사용되는 백신 생산 세포주",
            "interferon 생산 결핍 → 바이러스 증식에 유리",
            "다양한 바이러스 감수성 (광범위 숙주역)",
            "고밀도 배양 시 Gln 고갈 빠름",
        ],
        "nucleotide_adjustment": 0.7,  # high virus production
        "mw_preference": {"low": 1.1, "medium": 0.8, "high": 0.4},
    },

    "DF-1": {
        "name": "UMNSAH/DF-1 (Chicken Fibroblast)",
        "species": "Gallus gallus",
        "organism": "Chicken",
        "kegg_org": "gga",
        "tissue": "Embryo fibroblast",
        "application": "조류 바이러스 백신 (NDV, IBV, AI 등)",
        "base": "avian",  # key difference!
        "aa_adjustments": {
            # Avian-specific: Gly is semi-essential
            "Gly": 0.9,
            "Arg": 0.85,   # birds have higher Arg requirement
            "Gln": 0.75,
            "Pro": 0.7,    # collagen-rich fibroblast
        },
        "special_notes": [
            "조류 세포 → 포유류와 대사 경로 차이",
            "Gly 자체 합성 부족 (조류 특성)",
            "Arg 요구량 높음 (uric acid cycle)",
            "Vitamin C 자체 합성 가능 (L-gulonolactone oxidase 활성)",
            "조류 바이러스 백신의 핵심 숙주세포",
        ],
        "nucleotide_adjustment": 0.6,
        "mw_preference": {"low": 1.0, "medium": 0.8, "high": 0.5},
    },
}


# ── Public API ────────────────────────────────────────────────────────

def get_cell_line_prior(cell_line_id: str) -> dict[str, Any]:
    """Build complete demand profile for a cell line.

    Returns dict with keys:
        aa_demand: {AA_name: 0.0-1.0}
        vitamin_demand: {vitamin: 0.0-1.0}
        nucleotide_demand: float
        transporter_profile: {type: 0.0-1.0}
        mw_preference: {low/medium/high: weight}
        cell_line_info: metadata
        special_notes: list[str]
        source: str
    """
    if cell_line_id not in CELL_LINE_DB:
        # Unknown cell line → use base mammalian
        return {
            "aa_demand": dict(_BASE_MAMMALIAN["aa_demand"]),
            "vitamin_demand": dict(_BASE_MAMMALIAN["vitamin_demand"]),
            "nucleotide_demand": _BASE_MAMMALIAN["nucleotide_demand"],
            "transporter_profile": dict(_BASE_MAMMALIAN["transporter_profile"]),
            "mw_preference": {"low": 1.0, "medium": 0.8, "high": 0.5},
            "cell_line_info": {"name": cell_line_id},
            "special_notes": [],
            "source": "generic_mammalian",
        }

    info = CELL_LINE_DB[cell_line_id]
    base = _BASE_AVIAN if info["base"] == "avian" else _BASE_MAMMALIAN

    # Start from base profile
    aa_demand = dict(base["aa_demand"])

    # Apply cell-line-specific adjustments
    for aa, val in info.get("aa_adjustments", {}).items():
        aa_demand[aa] = val

    vitamin_demand = dict(base["vitamin_demand"])
    nuc_demand = info.get("nucleotide_adjustment", base["nucleotide_demand"])
    transporter = dict(base["transporter_profile"])
    mw_pref = info.get("mw_preference", {"low": 1.0, "medium": 0.8, "high": 0.5})

    return {
        "aa_demand": aa_demand,
        "vitamin_demand": vitamin_demand,
        "nucleotide_demand": nuc_demand,
        "transporter_profile": transporter,
        "mw_preference": mw_pref,
        "cell_line_info": {
            "name": info["name"],
            "species": info["species"],
            "organism": info["organism"],
            "kegg_org": info["kegg_org"],
            "tissue": info["tissue"],
            "application": info["application"],
        },
        "special_notes": info.get("special_notes", []),
        "source": "cell_line_literature",
    }


def get_all_cell_line_ids() -> list[str]:
    return list(CELL_LINE_DB.keys())


def get_cell_line_summary(cell_line_id: str) -> dict[str, Any]:
    """Quick summary of demand characteristics."""
    prior = get_cell_line_prior(cell_line_id)
    aa = prior["aa_demand"]

    high_demand = [k for k, v in sorted(aa.items(), key=lambda x: -x[1]) if v >= 0.8]
    medium_demand = [k for k, v in aa.items() if 0.4 <= v < 0.8]
    low_demand = [k for k, v in aa.items() if v < 0.4]

    return {
        "cell_line": cell_line_id,
        "high_demand_aa": high_demand,
        "medium_demand_aa": medium_demand,
        "low_demand_aa": low_demand,
        "vitamin_demand": "all B vitamins required",
        "nucleotide_demand": prior["nucleotide_demand"],
        "key_notes": prior["special_notes"][:3],
    }
