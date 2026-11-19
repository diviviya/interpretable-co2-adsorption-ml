"""
selection_service.py: the "Find materials" feature.

What it does, in plain words:
  The model predicts CO2 uptake from the properties of a FINISHED activated
  carbon (surface area, pore volumes, C/H/N/O/S) and the test conditions.
  So to compare biomass sources, we take real activated carbons that were
  made from each source and reported in published papers
  (artifacts/feedstock_carbons.json, every row cites its paper), and ask the
  model what each would adsorb at the customer's temperature and pressure.

  The same biomass gives very different carbons depending on how it is
  activated, so every feedstock shows several carbons, plus the spread.

Three functions are used by routers/selection.py:
  list_carbons()  the catalogue + sources + the model's training range
  screen()        rank all carbons (or one feedstock) at given conditions
  what_if()       one carbon with some properties changed
  sweep()         one carbon, one property varied across a range

Training-range problems are WARNINGS, not errors: the customer can still
look, but is told the number is an extrapolation. Only impossible input
(negative pressure, text instead of a number...) is refused.
"""
import json
import math

from ..config import ARTIFACTS
from .model_service import model_service

DATA_PATH = ARTIFACTS / "feedstock_carbons.json"

# short name used in the JSON / API  ->  exact column name the model expects
MATERIAL = {
    "surface_area": "Surface Area (m2/g)",
    "total_pore_volume": "Total Pore Volume(cm3/g)",
    "micropore_volume": "Micropore Volume (cm3/g)",
    "carbon": "C (%)",
    "hydrogen": "H (%)",
    "nitrogen": "N (%)",
    "oxygen": "O (%)",
    "sulfur": "S (%)",
}
TEMP = "Temp (°C)"
PRES = "Pressure (bar)"
PARAMS = {**MATERIAL, "temperature": TEMP, "pressure": PRES}

LABELS = {
    "surface_area": "Surface area",
    "total_pore_volume": "Total pore volume",
    "micropore_volume": "Micropore volume",
    "carbon": "Carbon",
    "hydrogen": "Hydrogen",
    "nitrogen": "Nitrogen",
    "oxygen": "Oxygen",
    "sulfur": "Sulfur",
    "temperature": "Temperature",
    "pressure": "Pressure",
}

# If one of these is missing in a paper, the model uses a typical value, so
# the result is flagged "partial". (Sulfur is left out: it is 0 for most carbons.)
IMPORTANT = ["total_pore_volume", "micropore_volume", "carbon", "hydrogen", "nitrogen", "oxygen"]

# Hard limits only stop nonsense. Being outside the training data is a warning.
T_LIMITS = (-50.0, 200.0)
P_LIMITS = (0.01, 50.0)
MAX_SWEEP_POINTS = 60

_cache: dict | None = None


def _data() -> dict:
    global _cache
    if _cache is None:
        if not DATA_PATH.exists():
            raise RuntimeError("artifacts/feedstock_carbons.json is missing.")
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        data["_by_id"] = {c["id"]: c for c in data["carbons"]}
        _cache = data
    return _cache


def _entry(carbon_id: str) -> dict:
    entry = _data()["_by_id"].get(carbon_id)
    if entry is None:
        raise KeyError(carbon_id)
    return entry


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------
def _num(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def _conditions(temperature, pressure) -> tuple[float, float]:
    t = _num(temperature, "Temperature")
    p = _num(pressure, "Pressure")
    if not T_LIMITS[0] <= t <= T_LIMITS[1]:
        raise ValueError(f"Temperature must be between {T_LIMITS[0]:g} and {T_LIMITS[1]:g} °C.")
    if not P_LIMITS[0] <= p <= P_LIMITS[1]:
        raise ValueError(f"Pressure must be between {P_LIMITS[0]:g} and {P_LIMITS[1]:g} bar.")
    return t, p


def _require_model():
    if model_service.bundle is None:
        raise RuntimeError("The model is not loaded yet. Try again in a moment.")


def _features(entry: dict, t: float, p: float, overrides: dict | None = None) -> dict:
    """The 10 model inputs for one carbon. None = not reported (the model fills
    in the typical training value, exactly as it does for blank form fields)."""
    feats = {MATERIAL[k]: entry.get(k) for k in MATERIAL}
    for key, value in (overrides or {}).items():
        if key not in MATERIAL:
            raise ValueError(f"'{key}' can't be changed. Choose one of: {', '.join(MATERIAL)}.")
        if value is None:
            feats[MATERIAL[key]] = None
            continue
        v = _num(value, LABELS[key])
        if v < 0:
            raise ValueError(f"{LABELS[key]} can't be negative.")
        feats[MATERIAL[key]] = v
    feats[TEMP], feats[PRES] = t, p
    return feats


def _range_warnings(values: dict) -> list[str]:
    """Which of these inputs lie outside what the model was trained on?"""
    info = model_service.feature_info or {}
    out = []
    for column, value in values.items():
        r = info.get(column)
        if value is None or not r:
            continue
        if value < r["min"] or value > r["max"]:
            out.append(
                f"{column} = {value:g} is outside the training data ({r['min']:g} to {r['max']:g}). "
                "The prediction is an extrapolation and much less reliable."
            )
    return out


def _predict(rows: list[dict]) -> list[tuple[float, dict | None]]:
    """(prediction, interval) for each row, all in one batch (fast)."""
    _require_model()
    preds = model_service.predict_many(rows)
    out = []
    for p in preds:
        p = float(p)
        interval = model_service.interval_for(p)
        if interval:
            interval = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in interval.items()}
        out.append((round(p, 4), interval))
    return out


def _verdict(target: float | None, prediction: float, interval: dict | None) -> str | None:
    """meets / borderline / unlikely.

    meets      - the best estimate reaches the target
    unlikely   - even the top of the likely range is below the target
    borderline - the best estimate is below the target but the range still reaches it
    (Whether the whole range clears the target is reported separately as "confident".)
    """
    if target is None:
        return None
    upper = interval["upper"] if interval else prediction
    if prediction >= target:
        return "meets"
    if upper < target:
        return "unlikely"
    return "borderline"


def _confident(target: float | None, prediction: float, interval: dict | None) -> bool | None:
    """True when even the low end of the likely range reaches the target."""
    if target is None:
        return None
    lower = interval["lower"] if interval else prediction
    return lower >= target


def _public(entry: dict) -> dict:
    missing = [LABELS[k] for k in IMPORTANT if entry.get(k) is None]
    return {
        "id": entry["id"],
        "feedstock": entry["feedstock"],
        "label": entry["label"],
        "activation": entry["activation"],
        "source": entry["source"],
        "surface_area": entry.get("surface_area"),
        "total_pore_volume": entry.get("total_pore_volume"),
        "micropore_volume": entry.get("micropore_volume"),
        "carbon": entry.get("carbon"),
        "hydrogen": entry.get("hydrogen"),
        "nitrogen": entry.get("nitrogen"),
        "oxygen": entry.get("oxygen"),
        "sulfur": entry.get("sulfur"),
        "reported_uptake_25c_1bar": entry.get("reported_uptake_25c_1bar"),
        "note": entry.get("note"),
        "quality": {"complete": not missing, "missing": missing},
    }


def _training_range() -> dict:
    info = model_service.feature_info or {}
    return {
        "temperature": info.get(TEMP),
        "pressure": info.get(PRES),
    }


# ---------------------------------------------------------------------
# 1) the catalogue
# ---------------------------------------------------------------------
def list_carbons() -> dict:
    data = _data()
    level = (model_service.conformal or {}).get("headline_level")
    return {
        "training_range": _training_range(),
        "interval_level": float(level) if level is not None else None,
        "feedstocks": [{"id": k, **v} for k, v in data["feedstocks"].items()],
        "sources": data["sources"],
        "carbons": [_public(c) for c in data["carbons"]],
    }


# ---------------------------------------------------------------------
# 2) rank carbons at the customer's conditions
# ---------------------------------------------------------------------
def screen(temperature, pressure, target_uptake=None, feedstock=None) -> dict:
    t, p = _conditions(temperature, pressure)
    target = None
    if target_uptake is not None:
        target = _num(target_uptake, "Target uptake")
        if target <= 0:
            raise ValueError("Target uptake must be greater than 0.")

    data = _data()
    wanted = None if feedstock in (None, "", "all") else feedstock
    if wanted is not None and wanted not in data["feedstocks"]:
        raise KeyError(wanted)
    carbons = [c for c in data["carbons"] if wanted is None or c["feedstock"] == wanted]

    results = _predict([_features(c, t, p) for c in carbons])
    rows = []
    for c, (pred, interval) in zip(carbons, results):
        rows.append({
            **_public(c),
            "predicted": pred,
            "interval": interval,
            "verdict": _verdict(target, pred, interval),
            "confident": _confident(target, pred, interval),
        })

    by_feedstock: dict[str, list[dict]] = {}
    for r in rows:
        by_feedstock.setdefault(r["feedstock"], []).append(r)

    groups = []
    for fid, items in by_feedstock.items():
        items.sort(key=lambda r: r["predicted"], reverse=True)
        best = items[0]
        groups.append({
            "id": fid,
            "name": data["feedstocks"][fid]["name"],
            "type": data["feedstocks"][fid]["type"],
            "carbons": items,
            "best": {k: best[k] for k in ("id", "label", "predicted", "interval", "verdict")},
            "spread": {"min": items[-1]["predicted"], "max": best["predicted"]},
        })
    groups.sort(key=lambda g: g["best"]["predicted"], reverse=True)

    return {
        "conditions": {"temperature": t, "pressure": p},
        "target_uptake": target,
        "warnings": _range_warnings({TEMP: t, PRES: p}),
        "feedstocks": groups,
    }


# ---------------------------------------------------------------------
# 3) change some properties of one carbon
# ---------------------------------------------------------------------
def what_if(carbon_id, temperature, pressure, overrides=None) -> dict:
    entry = _entry(carbon_id)
    t, p = _conditions(temperature, pressure)
    overrides = overrides or {}
    base = _features(entry, t, p)
    changed = _features(entry, t, p, overrides)
    (base_pred, base_int), (pred, interval) = _predict([base, changed])

    checked = {TEMP: t, PRES: p}
    for key in overrides:
        if overrides[key] is not None:
            checked[MATERIAL[key]] = changed[MATERIAL[key]]
    return {
        "carbon_id": carbon_id,
        "inputs": changed,
        "baseline": {"predicted": base_pred, "interval": base_int},
        "predicted": pred,
        "interval": interval,
        "change": round(pred - base_pred, 4),
        "warnings": _range_warnings(checked),
    }


# ---------------------------------------------------------------------
# 4) vary ONE property across a range and find the best value
# ---------------------------------------------------------------------
def sweep(carbon_id, parameter, temperature, pressure, overrides=None,
          start=None, stop=None, points=25) -> dict:
    entry = _entry(carbon_id)
    if parameter not in PARAMS:
        raise ValueError(f"Unknown parameter '{parameter}'. Choose one of: {', '.join(PARAMS)}.")
    column = PARAMS[parameter]
    t0, p0 = _conditions(temperature, pressure)

    info = (model_service.feature_info or {}).get(column) or {}
    lo = info.get("min") if start is None else start
    hi = info.get("max") if stop is None else stop
    if lo is None or hi is None:
        raise ValueError("Give a start and a stop value.")
    lo, hi = _num(lo, "Start"), _num(hi, "Stop")
    if hi <= lo:
        raise ValueError("Stop must be larger than start.")
    n = int(max(3, min(int(points), MAX_SWEEP_POINTS)))
    values = [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    rows = []
    for v in values:
        ov = dict(overrides or {})
        t, p = t0, p0
        if parameter == "temperature":
            t = v
        elif parameter == "pressure":
            p = v
        else:
            ov[parameter] = v
        t, p = _conditions(t, p)
        rows.append(_features(entry, t, p, ov))

    results = _predict(rows)
    pts = [{"value": round(v, 4), "predicted": r[0]} for v, r in zip(values, results)]
    best = max(pts, key=lambda x: x["predicted"])
    outside = bool(info) and (lo < info["min"] or hi > info["max"])
    return {
        "carbon_id": carbon_id,
        "parameter": parameter,
        "label": LABELS[parameter],
        "points": pts,
        "best": best,
        "training_range": {"min": info.get("min"), "max": info.get("max")} if info else None,
        "outside_training_range": outside,
    }
