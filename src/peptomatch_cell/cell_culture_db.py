"""SQLite-based cell culture database for PeptoMatch Cell.

Stores cell culture results (VCD, viability, IVCD, titer, μ, Qp time-series)
produced by cell-culture-app's processor.process_file(), and computes
aggregate metrics (max VCD, final viability, max titer, μ_max, ...) for the
recommendation / ML layers.

Schema mirrors the per-experiment / per-condition / per-metric layout
described in the v0.1 spec.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("peptomatch_cell.db")


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS culture_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_date TEXT,
    cell_line TEXT,
    culture_mode TEXT,
    title TEXT,
    basal_media TEXT,
    feed_media TEXT,
    feeding_days_json TEXT,
    source_filename TEXT,
    processed_at TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS culture_conditions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    treatment_name TEXT,
    treatment_raw TEXT,
    peptone_1 TEXT,
    peptone_2 TEXT,
    ratio_1 REAL,
    ratio_2 REAL,
    is_control INTEGER DEFAULT 0,
    FOREIGN KEY (experiment_id) REFERENCES culture_experiments(id)
);

CREATE TABLE IF NOT EXISTS culture_timeseries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id INTEGER NOT NULL,
    metric TEXT,
    unit TEXT,
    days_json TEXT,
    mean_json TEXT,
    std_json TEXT,
    replicates_json TEXT,
    FOREIGN KEY (condition_id) REFERENCES culture_conditions(id)
);

CREATE TABLE IF NOT EXISTS culture_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id INTEGER UNIQUE NOT NULL,
    max_vcd REAL,
    final_vcd REAL,
    final_viability REAL,
    min_viability REAL,
    ivcd_final REAL,
    max_titer REAL,
    final_titer REAL,
    max_qp REAL,
    mu_max REAL,
    combined_score REAL,
    computed_at TEXT,
    FOREIGN KEY (condition_id) REFERENCES culture_conditions(id)
);
"""


# ── Treatment name parsing ─────────────────────────────────────

_CONTROL_KEYWORDS = ("IMDM", "DMEM", "MEDIUM", "BASAL", "CONTROL", "MOCK")


def _parse_treatment(raw: str) -> dict[str, Any]:
    """Best-effort parse of a treatment label into peptone_1/2 + ratios.

    Handles common forms:
        "SOY-1"           → p1=SOY-1, ratio=100
        "SOY-1+RICE-1"    → p1=SOY-1, p2=RICE-1, 50/50
        "SOY-1 70+RICE 30"→ p1=SOY-1, ratio_1=70, p2=RICE, ratio_2=30
        "IMDM"            → control
    """
    s = (raw or "").strip()
    upper = s.upper()

    is_control = any(k in upper for k in _CONTROL_KEYWORDS) and "+" not in s
    if is_control:
        return {
            "treatment_name": s,
            "peptone_1": "",
            "peptone_2": "",
            "ratio_1": 0.0,
            "ratio_2": 0.0,
            "is_control": 1,
        }

    parts = [p.strip() for p in s.replace("/", "+").split("+") if p.strip()]
    if len(parts) >= 2:
        p1, p2 = parts[0], parts[1]
        # crude ratio extract (trailing number)
        import re
        def split_ratio(token: str) -> tuple[str, Optional[float]]:
            m = re.match(r"^(.+?)\s+(\d{1,3})(?:%)?\s*$", token)
            if m:
                return m.group(1).strip(), float(m.group(2))
            return token, None
        n1, r1 = split_ratio(p1)
        n2, r2 = split_ratio(p2)
        if r1 is None and r2 is None:
            r1, r2 = 50.0, 50.0
        elif r1 is None:
            r1 = max(0.0, 100.0 - (r2 or 0))
        elif r2 is None:
            r2 = max(0.0, 100.0 - r1)
        return {
            "treatment_name": s,
            "peptone_1": n1,
            "peptone_2": n2,
            "ratio_1": float(r1),
            "ratio_2": float(r2),
            "is_control": 0,
        }

    return {
        "treatment_name": s,
        "peptone_1": s,
        "peptone_2": "",
        "ratio_1": 100.0,
        "ratio_2": 0.0,
        "is_control": 0,
    }


# ── Metric helpers ─────────────────────────────────────────────

def _nan_safe_max(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(max(clean)) if clean else None


def _nan_safe_last(values: list) -> Optional[float]:
    for v in reversed(values or []):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    return None


def _nan_safe_min(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(min(clean)) if clean else None


class CellCultureDB:
    """SQLite persistence for cell-culture-app processing output."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else Path("data/cell_culture_data.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        self.conn.executescript(CREATE_TABLES_SQL)
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ── Ingest ──────────────────────────────────────────────────

    def ingest(self, payload: dict) -> dict:
        """Ingest a process_file() result.

        Expected payload (cell-culture-app/processor.py output):
            cell_line, culture_mode, feeding_days, title, sections{vcd,viability,ivcd,titer,mu,qp}
        Additional optional keys: source_filename, basal_media, feed_media,
        experiment_date, notes.
        """
        now = datetime.now().isoformat()
        cell_line = payload.get("cell_line") or ""
        culture_mode = payload.get("culture_mode") or ""
        title = payload.get("title") or f"{cell_line} {culture_mode}".strip()
        feeding_days = payload.get("feeding_days") or []
        source_filename = payload.get("source_filename") or payload.get("file_id") or "unknown"

        cur = self.conn.execute(
            """INSERT INTO culture_experiments
                (experiment_date, cell_line, culture_mode, title,
                 basal_media, feed_media, feeding_days_json,
                 source_filename, processed_at, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                payload.get("experiment_date") or now[:10],
                cell_line,
                culture_mode,
                title,
                payload.get("basal_media", "") or "",
                payload.get("feed_media", "") or "",
                json.dumps(feeding_days),
                source_filename,
                now,
                payload.get("notes", "") or "",
            ),
        )
        experiment_id = int(cur.lastrowid)

        sections = payload.get("sections") or {}

        # Collect all treatment names from every section
        treatment_set: list[str] = []
        for sec in sections.values():
            for name in (sec or {}).get("treatments", {}).keys():
                if name not in treatment_set:
                    treatment_set.append(name)

        # Insert conditions row per treatment
        condition_ids: dict[str, int] = {}
        for raw in treatment_set:
            parsed = _parse_treatment(raw)
            cur = self.conn.execute(
                """INSERT INTO culture_conditions
                    (experiment_id, treatment_name, treatment_raw,
                     peptone_1, peptone_2, ratio_1, ratio_2, is_control)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    parsed["treatment_name"],
                    raw,
                    parsed["peptone_1"],
                    parsed["peptone_2"],
                    parsed["ratio_1"],
                    parsed["ratio_2"],
                    parsed["is_control"],
                ),
            )
            condition_ids[raw] = int(cur.lastrowid)

        # Insert timeseries rows per (condition, metric)
        ts_rows = 0
        for metric, sec in sections.items():
            if not sec:
                continue
            unit = sec.get("unit", "")
            days = sec.get("days", [])
            for raw, stat in (sec.get("treatments") or {}).items():
                cid = condition_ids.get(raw)
                if cid is None:
                    continue
                self.conn.execute(
                    """INSERT INTO culture_timeseries
                        (condition_id, metric, unit,
                         days_json, mean_json, std_json, replicates_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cid,
                        metric,
                        unit,
                        json.dumps(days),
                        json.dumps(stat.get("mean", [])),
                        json.dumps(stat.get("std", [])),
                        json.dumps(stat.get("replicates", [])),
                    ),
                )
                ts_rows += 1

        self.conn.commit()

        self._compute_metrics_for_experiment(experiment_id)

        result = {
            "experiment_id": experiment_id,
            "conditions": len(condition_ids),
            "timeseries": ts_rows,
        }
        logger.info(f"Ingest OK: {result}")
        return result

    # ── Metrics ─────────────────────────────────────────────────

    def _compute_metrics_for_experiment(self, experiment_id: int) -> None:
        cur = self.conn.execute(
            "SELECT id FROM culture_conditions WHERE experiment_id = ?",
            (experiment_id,),
        )
        condition_ids = [int(r["id"]) for r in cur.fetchall()]

        for cid in condition_ids:
            try:
                metrics = self._compute_one(cid)
            except Exception as e:
                logger.warning(f"metric compute failed for condition {cid}: {e}")
                continue

            self.conn.execute(
                """INSERT OR REPLACE INTO culture_metrics
                    (condition_id, max_vcd, final_vcd, final_viability,
                     min_viability, ivcd_final, max_titer, final_titer,
                     max_qp, mu_max, combined_score, computed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cid,
                    metrics.get("max_vcd"),
                    metrics.get("final_vcd"),
                    metrics.get("final_viability"),
                    metrics.get("min_viability"),
                    metrics.get("ivcd_final"),
                    metrics.get("max_titer"),
                    metrics.get("final_titer"),
                    metrics.get("max_qp"),
                    metrics.get("mu_max"),
                    metrics.get("combined_score"),
                    datetime.now().isoformat(),
                ),
            )
        self.conn.commit()

    def _compute_one(self, condition_id: int) -> dict[str, Optional[float]]:
        cur = self.conn.execute(
            "SELECT metric, mean_json FROM culture_timeseries WHERE condition_id = ?",
            (condition_id,),
        )
        by_metric: dict[str, list] = {}
        for r in cur.fetchall():
            try:
                by_metric[r["metric"]] = json.loads(r["mean_json"] or "[]")
            except Exception:
                by_metric[r["metric"]] = []

        vcd = by_metric.get("vcd", [])
        via = by_metric.get("viability", [])
        ivcd = by_metric.get("ivcd", [])
        titer = by_metric.get("titer", [])
        qp = by_metric.get("qp", [])
        mu = by_metric.get("mu", [])

        max_vcd = _nan_safe_max(vcd)
        final_vcd = _nan_safe_last(vcd)
        final_viability = _nan_safe_last(via)
        min_viability = _nan_safe_min(via)
        ivcd_final = _nan_safe_last(ivcd)
        max_titer = _nan_safe_max(titer)
        final_titer = _nan_safe_last(titer)
        max_qp = _nan_safe_max(qp)
        mu_max = _nan_safe_max(mu)

        # Simple combined score: weighted log-ish blend of normalized signals.
        score_parts = []
        if max_vcd is not None:
            score_parts.append(("vcd", max_vcd, 1.0))
        if max_titer is not None:
            score_parts.append(("titer", max_titer / 100.0, 1.0))
        if final_viability is not None:
            score_parts.append(("viability", final_viability / 100.0, 0.5))
        if mu_max is not None:
            score_parts.append(("mu", mu_max, 0.5))
        if score_parts:
            total = sum(v * w for _k, v, w in score_parts)
            wsum = sum(w for _k, _v, w in score_parts)
            combined = total / wsum if wsum else None
        else:
            combined = None

        return {
            "max_vcd": max_vcd,
            "final_vcd": final_vcd,
            "final_viability": final_viability,
            "min_viability": min_viability,
            "ivcd_final": ivcd_final,
            "max_titer": max_titer,
            "final_titer": final_titer,
            "max_qp": max_qp,
            "mu_max": mu_max,
            "combined_score": combined,
        }

    # ── Queries ─────────────────────────────────────────────────

    def get_summary(self) -> dict:
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM culture_experiments")
        total_exp = int(cur.fetchone()["n"])
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM culture_conditions")
        total_cond = int(cur.fetchone()["n"])
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM culture_timeseries")
        total_ts = int(cur.fetchone()["n"])
        cur = self.conn.execute(
            "SELECT cell_line, COUNT(*) AS n FROM culture_experiments GROUP BY cell_line"
        )
        by_cell = {(r["cell_line"] or "Unknown"): int(r["n"]) for r in cur.fetchall()}
        cur = self.conn.execute(
            "SELECT culture_mode, COUNT(*) AS n FROM culture_experiments GROUP BY culture_mode"
        )
        by_mode = {(r["culture_mode"] or "Unknown"): int(r["n"]) for r in cur.fetchall()}
        return {
            "total_experiments": total_exp,
            "total_conditions": total_cond,
            "total_timeseries": total_ts,
            "by_cell_line": by_cell,
            "by_culture_mode": by_mode,
        }

    def get_experiments(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM culture_experiments ORDER BY id DESC"
        )
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            try:
                d["feeding_days"] = json.loads(d.pop("feeding_days_json") or "[]")
            except Exception:
                d["feeding_days"] = []
            rows.append(d)
        return rows

    def get_conditions_with_metrics(
        self, experiment_id: Optional[int] = None
    ) -> list[dict]:
        q = """
            SELECT c.*, m.max_vcd, m.final_vcd, m.final_viability, m.min_viability,
                   m.ivcd_final, m.max_titer, m.final_titer, m.max_qp, m.mu_max,
                   m.combined_score,
                   e.cell_line, e.culture_mode, e.title, e.basal_media, e.feed_media
              FROM culture_conditions c
              LEFT JOIN culture_metrics m ON m.condition_id = c.id
              JOIN culture_experiments e ON e.id = c.experiment_id
        """
        if experiment_id is not None:
            q += " WHERE c.experiment_id = ? ORDER BY c.id"
            cur = self.conn.execute(q, (int(experiment_id),))
        else:
            q += " ORDER BY c.id DESC"
            cur = self.conn.execute(q)
        return [dict(r) for r in cur.fetchall()]

    def get_timeseries(self, condition_id: int) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM culture_timeseries WHERE condition_id = ? ORDER BY id",
            (int(condition_id),),
        )
        out = []
        for r in cur.fetchall():
            d = dict(r)
            for k in ("days_json", "mean_json", "std_json", "replicates_json"):
                try:
                    d[k.replace("_json", "")] = json.loads(d.pop(k) or "[]")
                except Exception:
                    d[k.replace("_json", "")] = []
            out.append(d)
        return out

    def get_ml_training_data(self) -> list[dict]:
        """Flat condition-level rows suitable for downstream ML.

        Controls are kept (is_control=1) so callers can normalize against them.
        """
        cur = self.conn.execute(
            """SELECT c.id AS condition_id, c.treatment_name, c.treatment_raw,
                      c.peptone_1, c.peptone_2, c.ratio_1, c.ratio_2, c.is_control,
                      m.max_vcd, m.final_vcd, m.final_viability, m.min_viability,
                      m.ivcd_final, m.max_titer, m.final_titer, m.max_qp,
                      m.mu_max, m.combined_score,
                      e.id AS experiment_id, e.experiment_date, e.cell_line,
                      e.culture_mode, e.basal_media, e.feed_media,
                      e.feeding_days_json
                FROM culture_conditions c
                LEFT JOIN culture_metrics m ON m.condition_id = c.id
                JOIN culture_experiments e ON e.id = c.experiment_id
                ORDER BY c.id"""
        )
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            try:
                d["feeding_days"] = json.loads(d.pop("feeding_days_json") or "[]")
            except Exception:
                d["feeding_days"] = []
            rows.append(d)
        return rows

    def delete_experiment(self, experiment_id: int) -> dict:
        cur = self.conn.execute(
            "SELECT id FROM culture_conditions WHERE experiment_id = ?",
            (int(experiment_id),),
        )
        cond_ids = [int(r["id"]) for r in cur.fetchall()]
        n_ts = n_met = 0
        if cond_ids:
            placeholders = ",".join("?" * len(cond_ids))
            n_ts = self.conn.execute(
                f"DELETE FROM culture_timeseries WHERE condition_id IN ({placeholders})",
                cond_ids,
            ).rowcount
            n_met = self.conn.execute(
                f"DELETE FROM culture_metrics WHERE condition_id IN ({placeholders})",
                cond_ids,
            ).rowcount
        n_cond = self.conn.execute(
            "DELETE FROM culture_conditions WHERE experiment_id = ?",
            (int(experiment_id),),
        ).rowcount
        n_exp = self.conn.execute(
            "DELETE FROM culture_experiments WHERE id = ?",
            (int(experiment_id),),
        ).rowcount
        self.conn.commit()
        return {
            "experiments": int(n_exp),
            "conditions": int(n_cond),
            "timeseries": int(n_ts),
            "metrics": int(n_met),
        }

    def reset_all(self) -> dict:
        n_met = self.conn.execute("DELETE FROM culture_metrics").rowcount
        n_ts = self.conn.execute("DELETE FROM culture_timeseries").rowcount
        n_cond = self.conn.execute("DELETE FROM culture_conditions").rowcount
        n_exp = self.conn.execute("DELETE FROM culture_experiments").rowcount
        self.conn.execute(
            "DELETE FROM sqlite_sequence WHERE name IN "
            "('culture_experiments','culture_conditions','culture_timeseries','culture_metrics')"
        )
        self.conn.commit()
        return {
            "experiments": int(n_exp),
            "conditions": int(n_cond),
            "timeseries": int(n_ts),
            "metrics": int(n_met),
        }
