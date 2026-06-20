"""add pivot_escalation_count to missions

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-06-19
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("missions", sa.Column("pivot_escalation_count", sa.Integer(), nullable=True, server_default="0"))


def downgrade() -> None:
    op.drop_column("missions", "pivot_escalation_count")
