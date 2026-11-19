"""
selection.py: the URLs behind the "Find materials" page.

All of them need a logged-in account (same as Predict). The real work is in
services/selection_service.py; this file only turns requests into function
calls and turns problems into clear error messages.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..security import current_user
from ..services import selection_service as svc

router = APIRouter(prefix="/api/selection", dependencies=[Depends(current_user)])


class ScreenRequest(BaseModel):
    temperature: float = 25.0
    pressure: float = 1.0
    target_uptake: Optional[float] = None   # leave empty to just rank
    feedstock: Optional[str] = None         # leave empty (or "all") for every feedstock


class WhatIfRequest(BaseModel):
    carbon_id: str
    temperature: float = 25.0
    pressure: float = 1.0
    # properties to change, e.g. {"surface_area": 1800, "nitrogen": 4}
    overrides: dict[str, Optional[float]] = Field(default_factory=dict)


class SweepRequest(WhatIfRequest):
    parameter: str
    start: Optional[float] = None
    stop: Optional[float] = None
    points: int = 25


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as e:
        raise HTTPException(404, f"Unknown material or feedstock: {e.args[0]}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@router.get("/carbons")
def carbons():
    return _run(svc.list_carbons)


@router.post("/screen")
def screen(req: ScreenRequest):
    return _run(svc.screen, req.temperature, req.pressure, req.target_uptake, req.feedstock)


@router.post("/what-if")
def what_if(req: WhatIfRequest):
    return _run(svc.what_if, req.carbon_id, req.temperature, req.pressure, req.overrides)


@router.post("/sweep")
def sweep(req: SweepRequest):
    return _run(
        svc.sweep, req.carbon_id, req.parameter, req.temperature, req.pressure,
        req.overrides, req.start, req.stop, req.points,
    )
