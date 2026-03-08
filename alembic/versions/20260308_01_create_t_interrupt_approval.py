"""create t_interrupt_approval table

Revision ID: 20260308_01
Revises: 
Create Date: 2026-03-08 12:20:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260308_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "t_interrupt_approval",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=120), nullable=False),
        sa.Column("message_id", sa.String(length=200), nullable=False),
        sa.Column("action_name", sa.String(length=120), nullable=False),
        sa.Column("action_args", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("user_id", sa.BigInteger(), nullable=True),
        sa.Column("decision_time", sa.DateTime(), nullable=True),
        sa.Column("agent_name", sa.String(length=120), nullable=True),
        sa.Column("subgraph_thread_id", sa.String(length=180), nullable=True),
        sa.Column("checkpoint_id", sa.String(length=200), nullable=True),
        sa.Column("checkpoint_ns", sa.String(length=200), nullable=True),
        sa.Column("is_consumed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("create_time", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("update_time", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("session_id", "message_id", name="uq_interrupt_session_message"),
    )
    op.create_index("idx_interrupt_session_status_consumed", "t_interrupt_approval", ["session_id", "status", "is_consumed"])
    op.create_index("idx_interrupt_create_time", "t_interrupt_approval", ["create_time"])
    op.create_index(op.f("ix_t_interrupt_approval_session_id"), "t_interrupt_approval", ["session_id"])
    op.create_index(op.f("ix_t_interrupt_approval_message_id"), "t_interrupt_approval", ["message_id"])
    op.create_index(op.f("ix_t_interrupt_approval_status"), "t_interrupt_approval", ["status"])
    op.create_index(op.f("ix_t_interrupt_approval_user_id"), "t_interrupt_approval", ["user_id"])
    op.create_index(op.f("ix_t_interrupt_approval_is_consumed"), "t_interrupt_approval", ["is_consumed"])


def downgrade() -> None:
    op.drop_index(op.f("ix_t_interrupt_approval_is_consumed"), table_name="t_interrupt_approval")
    op.drop_index(op.f("ix_t_interrupt_approval_user_id"), table_name="t_interrupt_approval")
    op.drop_index(op.f("ix_t_interrupt_approval_status"), table_name="t_interrupt_approval")
    op.drop_index(op.f("ix_t_interrupt_approval_message_id"), table_name="t_interrupt_approval")
    op.drop_index(op.f("ix_t_interrupt_approval_session_id"), table_name="t_interrupt_approval")
    op.drop_index("idx_interrupt_create_time", table_name="t_interrupt_approval")
    op.drop_index("idx_interrupt_session_status_consumed", table_name="t_interrupt_approval")
    op.drop_table("t_interrupt_approval")
