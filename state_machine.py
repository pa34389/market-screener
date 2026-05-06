"""DAG-style setup states, scoring, and grade — deterministic (no LLM)."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

DagState = Literal[
    "NO_SETUP",
    "HTF_BIAS_FOUND",
    "RANGE_DEFINED",
    "POI_IDENTIFIED",
    "WAITING_FOR_POI",
    "PRICE_INSIDE_POI",
    "LIQUIDITY_SWEEP_SEEN",
    "LTF_MSB_CONFIRMED",
    "BREAKER_RETEST_PENDING",
    "SETUP_VALID_FOR_REVIEW",
    "INVALID_RR",
    "INVALIDATED",
]


def grade_from_score(score: int) -> str:
    if score >= 85:
        return "A-grade setup"
    if score >= 70:
        return "B-grade setup"
    if score >= 55:
        return "Watch only"
    return "Ignore"


def compute_score_components(
    *,
    msb_found: bool,
    bias_neutral: bool,
    valid_range: bool,
    price_valid_half: bool,
    poi_found: bool,
    inside_or_near_poi: bool,
    liquidity_sweep: bool,
    ltf_msb: bool,
    rr_ok: bool,
    engineered: bool,
    displacement_hint: bool,
) -> int:
    """Spec §12 additive components, normalized to 0–100 (raw max 120)."""
    raw = 0
    if msb_found and not bias_neutral:
        raw += 15
    if msb_found:
        raw += 15
    if valid_range:
        raw += 10
    if price_valid_half:
        raw += 15
    if poi_found:
        raw += 15
    if inside_or_near_poi:
        raw += 10
    if liquidity_sweep:
        raw += 10
    if ltf_msb:
        raw += 10
    if rr_ok:
        raw += 10
    if displacement_hint:
        raw += 10
    if engineered:
        raw += 5
    return min(100, int(round(raw * 100 / 120)))


def compute_dag_state(
    *,
    invalidated: bool,
    htf_msb: bool,
    bias_neutral: bool,
    valid_range: bool,
    price_valid_half: bool,
    poi_found: bool,
    inside_poi: bool,
    near_poi: bool,
    liquidity_sweep: bool,
    ltf_msb: bool,
    rr: Optional[float],
    min_rr: float,
    breaker_retest_seen: bool,
) -> DagState:
    if invalidated:
        return "INVALIDATED"
    if not htf_msb or bias_neutral:
        return "NO_SETUP"
    if not valid_range:
        return "HTF_BIAS_FOUND"
    if not price_valid_half:
        return "RANGE_DEFINED"
    if not poi_found:
        return "POI_IDENTIFIED"
    inside_or_near = inside_poi or near_poi
    if not inside_or_near:
        return "WAITING_FOR_POI"
    if not liquidity_sweep:
        return "PRICE_INSIDE_POI"
    if not ltf_msb:
        return "LIQUIDITY_SWEEP_SEEN"
    if rr is None or rr < min_rr:
        return "INVALID_RR"
    if not breaker_retest_seen:
        return "BREAKER_RETEST_PENDING"
    return "SETUP_VALID_FOR_REVIEW"


def missing_conditions_for_state(
    dag: DagState,
    *,
    htf_msb: bool,
    valid_range: bool,
    price_valid_half: bool,
    poi_found: bool,
    inside_or_near_poi: bool,
    liquidity_sweep: bool,
    ltf_msb: bool,
    rr: Optional[float],
    min_rr: float,
    breaker_retest_seen: bool,
) -> List[str]:
    out: List[str] = []
    if dag == "INVALIDATED":
        return ["Setup invalidated vs stop level"]
    if not htf_msb:
        out.append("HTF market structure break (close beyond swing)")
    if htf_msb and not valid_range:
        out.append("Dealing range not defined")
    if valid_range and not price_valid_half:
        out.append("Price not in correct premium/discount for bias")
    if price_valid_half and not poi_found:
        out.append("Identify HTF POI in correct range half")
    if poi_found and not inside_or_near_poi:
        out.append("Price at or inside HTF POI zone")
    if inside_or_near_poi and not liquidity_sweep:
        out.append("LTF liquidity sweep (wick through, close back inside)")
    if liquidity_sweep and not ltf_msb:
        out.append("LTF MSB after sweep (close confirmation)")
    if ltf_msb and (rr is None or rr < min_rr):
        out.append(f"RR >= {min_rr} using sweep invalidation and external target")
    if ltf_msb and rr is not None and rr >= min_rr and not breaker_retest_seen:
        out.append("Breaker / retest interaction on LTF")
    return out


def should_alert_dag(dag: DagState) -> bool:
    return dag in {
        "PRICE_INSIDE_POI",
        "LIQUIDITY_SWEEP_SEEN",
        "BREAKER_RETEST_PENDING",
        "SETUP_VALID_FOR_REVIEW",
        "INVALID_RR",
        "INVALIDATED",
    }


def legacy_status_from_dag(dag: DagState) -> str:
    """Map DAG state to compact UI status (legacy five-way label)."""
    if dag in ("NO_SETUP", "HTF_BIAS_FOUND", "POI_IDENTIFIED", "INVALIDATED"):
        return "invalid"
    if dag in ("RANGE_DEFINED", "WAITING_FOR_POI", "INVALID_RR"):
        return "watch"
    if dag == "PRICE_INSIDE_POI":
        return "poi_entered"
    if dag == "LIQUIDITY_SWEEP_SEEN":
        return "sweep_seen"
    if dag in ("LTF_MSB_CONFIRMED", "BREAKER_RETEST_PENDING"):
        return "sweep_seen"
    if dag == "SETUP_VALID_FOR_REVIEW":
        return "confirmed"
    return "invalid"
