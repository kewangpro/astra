"""add best_metric_iteration and current_metric_value to missions

Revision ID: a1b2c3d4e5f6
Revises: f3a9b2c1d8e7
Create Date: 2026-06-18
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f6"
down_revision = "f3a9b2c1d8e7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("missions", sa.Column("best_metric_iteration", sa.Integer(), nullable=True))
    op.add_column("missions", sa.Column("current_metric_value", sa.String(100), nullable=True))


def downgrade() -> None:
    op.drop_column("missions", "current_metric_value")
    op.drop_column("missions", "best_metric_iteration")
