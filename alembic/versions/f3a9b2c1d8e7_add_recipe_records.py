"""add_recipe_records

Revision ID: f3a9b2c1d8e7
Revises: d8199b5e6752
Create Date: 2026-06-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f3a9b2c1d8e7'
down_revision: Union[str, Sequence[str], None] = 'd8199b5e6752'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'recipe_records',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('domain', sa.String(100), nullable=False),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('hyperparameters', sa.JSON(), nullable=True),
        sa.Column('curriculum', sa.JSON(), nullable=True),
        sa.Column('reward_shaping', sa.JSON(), nullable=True),
        sa.Column('full_content', sa.JSON(), nullable=True),
        sa.Column('mission_id', sa.String(36), nullable=True),
        sa.Column('parent_recipe_id', sa.String(36), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('target_metric', sa.JSON(), nullable=True),
        sa.Column('generation', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('consecutive_wins', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_golden', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_recipe_records_name', 'recipe_records', ['name'], unique=True)
    op.create_index('ix_recipe_records_domain', 'recipe_records', ['domain'], unique=False)
    op.create_index('ix_recipe_records_mission_id', 'recipe_records', ['mission_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_recipe_records_mission_id', table_name='recipe_records')
    op.drop_index('ix_recipe_records_domain', table_name='recipe_records')
    op.drop_index('ix_recipe_records_name', table_name='recipe_records')
    op.drop_table('recipe_records')
