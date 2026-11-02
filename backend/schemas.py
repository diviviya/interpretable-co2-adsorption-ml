
from typing import Any
import math
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=10, max_length=128)

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

        # Physical/domain validation shared by all single-prediction requests.
        non_negative = {
            "surface_area": "Surface area",
            "total_pore_volume": "Total pore volume",
            "micropore_volume": "Micropore volume",
        }
        for field_name, label in non_negative.items():
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{label} cannot be negative.")

        percentages = {
            "carbon": "Carbon",
            "hydrogen": "Hydrogen",
            "nitrogen": "Nitrogen",
            "oxygen": "Oxygen",
            "sulfur": "Sulfur",
        }
        for field_name, label in percentages.items():
            value = getattr(self, field_name)
            if value is not None and not 0 <= value <= 100:
                raise ValueError(f"{label} percentage must be between 0 and 100.")

        temperature = self.temperature
        if temperature is not None and temperature <= -273.15:
            raise ValueError("Temperature must be above absolute zero (-273.15 °C).")

        pressure = self.pressure
        if pressure is not None and pressure <= 0:
            raise ValueError("Pressure must be greater than 0 bar.")

        return self

class WhatIfRequest(BaseModel):
    baseline: FeatureInput
    modified: FeatureInput

class TrainJobStartRequest(BaseModel):
    dataset_name: str = Field(min_length=1, max_length=255)

# -----------------------------------------------------------------------------
# GENERIC RESPONSE SCHEMAS
# -----------------------------------------------------------------------------
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Any | None = None
