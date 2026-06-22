"""add best_policy_kwargs to missions

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-06-21
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("missions", sa.Column("best_policy_kwargs", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("missions", "best_policy_kwargs")
