"""All database access lives here – routers never touch the session directly."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import TravelPlan, User


# ----------------------------------------------------------------------
# Users
# ----------------------------------------------------------------------
def get_user_by_supabase_id(db: Session, supabase_user_id: str) -> User | None:
    return db.scalar(select(User).where(User.supabase_user_id == supabase_user_id))


def get_or_create_user(
    db: Session,
    *,
    supabase_user_id: str,
    email: str | None = None,
    display_name: str | None = None,
    avatar_url: str | None = None,
) -> User:
    """Return the local user for a Supabase subject, creating it once."""
    user = get_user_by_supabase_id(db, supabase_user_id)
    if user is not None:
        return user

    user = User(
        supabase_user_id=supabase_user_id,
        email=email,
        display_name=display_name,
        avatar_url=avatar_url,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # Concurrent first request – another worker won the race.
        db.rollback()
        existing = get_user_by_supabase_id(db, supabase_user_id)
        if existing is None:
            raise
        return existing

    db.refresh(user)
    return user


def update_user(db: Session, user: User, data: dict[str, Any]) -> User:
    if not data:
        return user
    for field, value in data.items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return user


# ----------------------------------------------------------------------
# Travel plans
# ----------------------------------------------------------------------
def list_travel_plans(
    db: Session, *, user_id: uuid.UUID, limit: int = 50, offset: int = 0
) -> list[TravelPlan]:
    stmt = (
        select(TravelPlan)
        .where(TravelPlan.user_id == user_id)
        .order_by(TravelPlan.created_at.desc(), TravelPlan.id)
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(stmt).all())


def get_travel_plan(
    db: Session, *, user_id: uuid.UUID, plan_id: uuid.UUID
) -> TravelPlan | None:
    """Ownership is part of the query – never a separate check."""
    return db.scalar(
        select(TravelPlan).where(
            TravelPlan.id == plan_id,
            TravelPlan.user_id == user_id,
        )
    )


def create_travel_plan(
    db: Session, *, user_id: uuid.UUID, data: dict[str, Any]
) -> TravelPlan:
    plan = TravelPlan(user_id=user_id, **data)
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


def update_travel_plan(db: Session, plan: TravelPlan, data: dict[str, Any]) -> TravelPlan:
    if not data:
        return plan
    for field, value in data.items():
        setattr(plan, field, value)
    db.commit()
    db.refresh(plan)
    return plan


def delete_travel_plan(db: Session, plan: TravelPlan) -> None:
    db.delete(plan)
    db.commit()
