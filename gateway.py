"""PeptoMatch Cell FastAPI gateway (Railway-ready).

Server-rendered HTML pages + JSON API for the cell-line peptone
recommendation demo. Mirrors the architecture of peptomatch/gateway.py
but is dedicated to animal cell lines.

Endpoints
─────────
UI pages
    GET  /                  Dashboard (cell-line summary + culture experiments)
    GET  /recommend         Peptone recommendation form + results
    GET  /culture           Cell culture experiment browser

Cell Culture Data API (called by cell-culture-app)
    POST /api/ingest
    GET  /api/cell/summary
    GET  /api/cell/experiments
    GET  /api/cell/conditions
    GET  /api/cell/timeseries
    GET  /api/cell/ml-data
    DELETE /api/cell/experiments/{id}
    DELETE /api/cell/reset

Recommendation API
    POST /api/recommend     JSON in/out, used by /recommend page

Health
    GET  /healthz
    GET  /api/health
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger("peptomatch_cell.gateway")
logging.basicConfig(level=logging.INFO)

# ── Path setup ───────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
_SRC = _HERE / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Backend imports (wrapped so /healthz still works on partial failures)
try:
    from peptomatch_cell.cell_culture_db import CellCultureDB
except Exception as e:
    logger.error(f"Failed to import CellCultureDB: {e}")
    CellCultureDB = None  # type: ignore

try:
    from peptomatch_cell.utils import load_config
    from peptomatch_cell.io_loaders import load_composition_data
    from peptomatch_cell.composition_features import CompositionFeatureExtractor
    from peptomatch_cell.scoring import CellPeptoneRecommender
    from peptomatch_cell.cell_line_priors import (
        CELL_LINE_DB, get_all_cell_line_ids, get_cell_line_summary,
    )
    _BACKEND_OK = True
except Exception as e:
    logger.error(f"Failed to import peptomatch_cell backend: {e}")
    _BACKEND_OK = False


# ── Globals populated by lifespan ────────────────────────────
cell_db: Optional["CellCultureDB"] = None
comp_df: Optional[Any] = None
extractor: Optional[Any] = None
recommender: Optional[Any] = None
app_config: Optional[dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cell_db, comp_df, extractor, recommender, app_config

    if CellCultureDB is not None:
        try:
            db_path = os.getenv(
                "CELL_CULTURE_DB_PATH",
                str(_HERE / "data" / "cell_culture_data.db"),
            )
            cell_db = CellCultureDB(Path(db_path))
            logger.info(f"CellCultureDB ready: {cell_db.db_path}")
        except Exception as e:
            logger.error(f"CellCultureDB init failed: {e}")

    if _BACKEND_OK:
        try:
            cfg_path = _HERE / "config" / "config.yaml"
            app_config = load_config(cfg_path) if cfg_path.exists() else load_config()
            logger.info("config.yaml loaded")
        except Exception as e:
            logger.error(f"config load failed: {e}")

        try:
            comp_path = _HERE / "data" / "composition_template.xlsx"
            comp_df = load_composition_data(comp_path)
            extractor = CompositionFeatureExtractor(comp_df)
            recommender = CellPeptoneRecommender(
                extractor,
                (app_config or {}).get("weights"),
            )
            logger.info(f"composition + recommender ready ({len(comp_df)} peptones)")
        except Exception as e:
            logger.error(f"recommender init failed: {e}")

    logger.info("PeptoMatch Cell gateway startup complete")
    yield

    if cell_db is not None:
        cell_db.close()


app = FastAPI(title="PeptoMatch Cell", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _unhandled_exc(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Unhandled error on {request.url.path}:\n{tb}")
    debug = os.getenv("PEPTOMATCH_DEBUG", "1") == "1"
    if debug:
        body = (
            f"<html><body style='font-family:monospace;background:#0b1020;color:#f87171;"
            f"padding:20px;'><h2>500 on {request.url.path}</h2><pre>{tb}</pre></body></html>"
        )
        return HTMLResponse(body, status_code=500)
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Templates + static ───────────────────────────────────────
TEMPLATES_DIR = _HERE / "templates"
STATIC_DIR = _HERE / "static"
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _ctx(request: Request, **extra) -> dict:
    return {
        "request": request,
        "backend_ok": _BACKEND_OK and recommender is not None,
        "cell_db_ok": cell_db is not None,
        **extra,
    }


# ── Health ──────────────────────────────────────────────────

@app.get("/healthz")
async def healthz():
    return JSONResponse({
        "status": "ok",
        "db_ready": cell_db is not None,
        "recommender_ready": recommender is not None,
    })


@app.get("/api/health")
async def api_health():
    return JSONResponse({"status": "ok"})


# ── UI pages ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    summary: dict = {}
    if cell_db is not None:
        try:
            summary = cell_db.get_summary()
        except Exception as e:
            logger.warning(f"summary failed: {e}")

    cell_lines = []
    if _BACKEND_OK:
        try:
            for cl_id in get_all_cell_line_ids():
                info = CELL_LINE_DB.get(cl_id, {})
                cell_lines.append({
                    "id": cl_id,
                    "name": info.get("name", cl_id),
                    "application": info.get("application", ""),
                })
        except Exception:
            pass

    peptone_count = int(len(comp_df)) if comp_df is not None else 0

    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context=_ctx(
            request,
            summary=summary,
            cell_lines=cell_lines,
            peptone_count=peptone_count,
        ),
    )


@app.get("/recommend", response_class=HTMLResponse)
async def recommend_page(request: Request):
    cell_lines = []
    if _BACKEND_OK:
        for cl_id in get_all_cell_line_ids():
            info = CELL_LINE_DB.get(cl_id, {})
            cell_lines.append({"id": cl_id, "name": info.get("name", cl_id)})
    return templates.TemplateResponse(
        request=request,
        name="recommend.html",
        context=_ctx(request, cell_lines=cell_lines),
    )


@app.get("/culture", response_class=HTMLResponse)
async def culture_page(request: Request):
    experiments: list[dict] = []
    summary: dict = {}
    if cell_db is not None:
        try:
            experiments = cell_db.get_experiments()
            summary = cell_db.get_summary()
        except Exception as e:
            logger.warning(f"culture page query failed: {e}")
    return templates.TemplateResponse(
        request=request,
        name="culture.html",
        context=_ctx(request, experiments=experiments, summary=summary),
    )


# ── Recommendation API ─────────────────────────────────────

@app.post("/api/recommend")
async def api_recommend(payload: dict):
    if recommender is None:
        return JSONResponse(
            status_code=503,
            content={"error": "recommender not initialized"},
        )
    try:
        cell_line_id = payload.get("cell_line_id") or payload.get("cell_line")
        if not cell_line_id:
            return JSONResponse(status_code=400, content={"error": "cell_line_id required"})
        top_k = int(payload.get("top_k", 10))
        sempio_only = bool(payload.get("sempio_only", True))

        sample_names = None
        if sempio_only and app_config:
            pf = app_config.get("peptone_filter") or []
            avail = set(comp_df["Sample_name"].tolist())
            sample_names = [n for n in pf if n in avail]

        results = recommender.recommend(
            cell_line_id, sample_names=sample_names, top_k=top_k,
        )
        summary = get_cell_line_summary(cell_line_id) if _BACKEND_OK else {}

        return JSONResponse({
            "status": "ok",
            "cell_line": cell_line_id,
            "summary": summary,
            "recommendations": [
                {
                    "rank": r.rank,
                    "sample_name": r.sample_name,
                    "raw_material": r.raw_material,
                    "material_type": r.material_type,
                    "total_score": round(r.total_score, 2),
                    "sub_scores": {k: round(v, 3) for k, v in r.sub_scores.items()},
                    "strengths": r.strengths,
                    "weaknesses": r.weaknesses,
                }
                for r in results
            ],
        })
    except Exception as e:
        logger.exception(f"recommend failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Cell Culture Data API ───────────────────────────────────

def _require_db():
    if cell_db is None:
        return JSONResponse(status_code=503, content={"error": "cell_db not initialized"})
    return None


@app.post("/api/ingest")
async def ingest_culture_data(payload: dict):
    err = _require_db()
    if err:
        return err
    try:
        result = cell_db.ingest(payload)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        logger.exception(f"ingest failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@app.get("/api/cell/summary")
async def api_cell_summary():
    err = _require_db()
    if err:
        return err
    return JSONResponse(cell_db.get_summary())


@app.get("/api/cell/experiments")
async def api_cell_experiments():
    err = _require_db()
    if err:
        return err
    return JSONResponse(cell_db.get_experiments())


@app.get("/api/cell/conditions")
async def api_cell_conditions(experiment_id: Optional[int] = None):
    err = _require_db()
    if err:
        return err
    return JSONResponse(cell_db.get_conditions_with_metrics(experiment_id))


@app.get("/api/cell/timeseries")
async def api_cell_timeseries(condition_id: int):
    err = _require_db()
    if err:
        return err
    return JSONResponse(cell_db.get_timeseries(condition_id))


@app.get("/api/cell/ml-data")
async def api_cell_ml_data():
    err = _require_db()
    if err:
        return err
    return JSONResponse(cell_db.get_ml_training_data())


@app.delete("/api/cell/experiments/{experiment_id}")
async def api_delete_experiment(experiment_id: int):
    err = _require_db()
    if err:
        return err
    try:
        result = cell_db.delete_experiment(experiment_id)
        if result["experiments"] == 0:
            return JSONResponse(
                status_code=404,
                content={"status": "not_found", "experiment_id": experiment_id},
            )
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@app.delete("/api/cell/reset")
async def api_reset(confirm: str = ""):
    err = _require_db()
    if err:
        return err
    if confirm != "yes":
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "confirm=yes required"},
        )
    return JSONResponse({"status": "ok", **cell_db.reset_all()})
