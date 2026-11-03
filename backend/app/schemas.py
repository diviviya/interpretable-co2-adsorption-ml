"""
schemas.py: the shape of data going in and out of the API.

FastAPI uses these Pydantic classes to check every request automatically.
If a request doesn't match, FastAPI replies 422 with a "detail" message,
and the frontend shows that message to the user.
"""
from typing import Optional

from pydantic import BaseModel, field_validator

from .services.model_service import is_number

# Inputs the user must fill in. The two pore volumes may be left blank
# (the frontend marks them optional too); the model then uses the median.
REQUIRED = {
    "Surface Area (m2/g)", "C (%)", "H (%)", "N (%)", "O (%)", "S (%)",
    "Temp (°C)", "Pressure (bar)",
}
OPTIONAL = {"Total Pore Volume(cm3/g)", "Micropore Volume (cm3/g)"}
ALL_FEATURES = REQUIRED | OPTIONAL
CAN_BE_NEGATIVE = {"Temp (°C)"}


class PredictRequest(BaseModel):
    features: dict[str, Optional[float]]

    @field_validator("features")
    @classmethod
    def check_features(cls, v: dict) -> dict:
        unknown = set(v) - ALL_FEATURES
        if unknown:
            raise ValueError(f"Unknown input(s): {', '.join(sorted(unknown))}")

        missing = [f for f in sorted(REQUIRED) if v.get(f) is None]
        if missing:
            raise ValueError(f"Missing required input(s): {', '.join(missing)}")

        for name, value in v.items():
            if value is None:
                continue
            if not is_number(value):
                raise ValueError(f"{name} must be a finite number")
            if value < 0 and name not in CAN_BE_NEGATIVE:
                raise ValueError(f"{name} cannot be negative")

        # Optional inputs that weren't sent at all count as blank.
        return {f: v.get(f) for f in ALL_FEATURES}


class ShapItem(BaseModel):
    feature: str
    value: Optional[float]
    contribution: float


class PredictResponse(BaseModel):
    prediction: float
    unit: str = "mmol/g"
    model_version: str
    base_value: Optional[float] = None
    shap: list[ShapItem] = []
    explanation: Optional[str] = None
    warnings: list[str] = []
