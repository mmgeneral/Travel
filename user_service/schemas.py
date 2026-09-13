"""Pydantic v2 request / response models.

``extra="forbid"`` everywhere means clients cannot smuggle in ``user_id``
or any other unexpected field.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

PlanStatus = Literal["draft", "planned", "booked", "completed", "archived"]


# ----------------------------------------------------------------------
# Users
# ----------------------------------------------------------------------
class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    supabase_user_id: str
    email: str | None = None
    display_name: str | None = None
    avatar_url: str | None = None
    locale: str | None = None
    created_at: datetime
    updated_at: datetime


class UserUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=255)
    avatar_url: str | None = None
    locale: str | None = Field(default=None, max_length=35)


# ----------------------------------------------------------------------
# Travel plans
# ----------------------------------------------------------------------
class TravelPlanCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=255)
    destination: str | None = Field(default=None, max_length=255)
    start_date: date | None = None
    end_date: date | None = None
    status: PlanStatus | None = None
    content: dict[str, Any] = Field(default_factory=dict)


class TravelPlanUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=255)
    destination: str | None = Field(default=None, max_length=255)
    start_date: date | None = None
    end_date: date | None = None
    status: PlanStatus | None = None
    content: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _reject_null_required_columns(self) -> "TravelPlanUpdate":
        for field in ("title", "content"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self


class TravelPlanRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    title: str
    destination: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    status: str | None = None
    content: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str
    service: str
