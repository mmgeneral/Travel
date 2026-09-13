"""``/api/v1/travel-plans`` endpoints.

Every lookup is scoped by the authenticated user id. A plan that belongs to
somebody else is indistinguishable from a missing plan (404, never 403).
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from .. import crud
from ..database import get_db
from ..deps import get_current_user
from ..models import TravelPlan, User
from ..schemas import TravelPlanCreate, TravelPlanRead, TravelPlanUpdate

router = APIRouter(prefix="/api/v1/travel-plans", tags=["travel-plans"])

_NOT_FOUND = "Travel plan not found"


def _get_owned_plan(db: Session, user: User, plan_id: UUID) -> TravelPlan:
    plan = crud.get_travel_plan(db, user_id=user.id, plan_id=plan_id)
    if plan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    return plan


@router.get("", response_model=list[TravelPlanRead])
def list_travel_plans(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[TravelPlan]:
    return crud.list_travel_plans(
        db, user_id=current_user.id, limit=limit, offset=offset
    )


@router.post("", response_model=TravelPlanRead, status_code=status.HTTP_201_CREATED)
def create_travel_plan(
    payload: TravelPlanCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TravelPlan:
    return crud.create_travel_plan(
        db, user_id=current_user.id, data=payload.model_dump()
    )


@router.get("/{plan_id}", response_model=TravelPlanRead)
def read_travel_plan(
    plan_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TravelPlan:
    return _get_owned_plan(db, current_user, plan_id)


@router.patch("/{plan_id}", response_model=TravelPlanRead)
def update_travel_plan(
    plan_id: UUID,
    payload: TravelPlanUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TravelPlan:
    plan = _get_owned_plan(db, current_user, plan_id)
    data = payload.model_dump(exclude_unset=True)
    return crud.update_travel_plan(db, plan, data)


@router.delete("/{plan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_travel_plan(
    plan_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    plan = _get_owned_plan(db, current_user, plan_id)
    crud.delete_travel_plan(db, plan)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
