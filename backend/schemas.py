
from typing import Any
import math
from pydantic import BaseModel, ConfigDict, model_validator


class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "user"

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class UserResponse(BaseModel):
    id: int
    username: str
    role: str


class FeatureInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    surface_area: float | None = None
    total_pore_volume: float | None = None
    micropore_volume: float | None = None
    carbon: float | None = None
    carbon_pct: float | None = None
    hydrogen: float | None = None
    hydrogen_pct: float | None = None
    nitrogen: float | None = None
    nitrogen_pct: float | None = None
    oxygen: float | None = None
    oxygen_pct: float | None = None
    sulfur: float | None = None
    sulfur_pct: float | None = None
    temperature: float | None = None
    temperature_c: float | None = None
    pressure: float | None = None
    pressure_bar: float | None = None

    @model_validator(mode="after")
    def normalize_fields(self):
        if self.carbon is None and self.carbon_pct is not None:
            self.carbon = self.carbon_pct
        if self.hydrogen is None and self.hydrogen_pct is not None:
            self.hydrogen = self.hydrogen_pct
        if self.nitrogen is None and self.nitrogen_pct is not None:
            self.nitrogen = self.nitrogen_pct
        if self.oxygen is None and self.oxygen_pct is not None:
            self.oxygen = self.oxygen_pct
        if self.sulfur is None and self.sulfur_pct is not None:
            self.sulfur = self.sulfur_pct
        if self.temperature is None and self.temperature_c is not None:
            self.temperature = self.temperature_c
        if self.pressure is None and self.pressure_bar is not None:
            self.pressure = self.pressure_bar

        for name, value in self:
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number or null")
        return self

class WhatIfRequest(BaseModel):
    baseline: FeatureInput
    modified: FeatureInput

class TrainJobStartRequest(BaseModel):
    dataset_name: str = "Processed_527_datapoints.csv"

# -----------------------------------------------------------------------------
# GENERIC RESPONSE SCHEMAS
# -----------------------------------------------------------------------------
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Any | None = None
