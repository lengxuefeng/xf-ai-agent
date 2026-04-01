"""add user skill and update mcp

Revision ID: aab1c2ea9450
Revises: 20260308_01
Create Date: 2026-04-01 10:42:48.511823

"""
import json

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'aab1c2ea9450'
down_revision = '20260308_01'
branch_labels = None
depends_on = None


def _has_table(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_column(inspector, table_name: str, column_name: str) -> bool:
    if not _has_table(inspector, table_name):
        return False
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    if not _has_table(inspector, table_name):
        return False
    return any(index["name"] == index_name for index in inspector.get_indexes(table_name))


def _has_check_constraint(inspector, table_name: str, constraint_name: str) -> bool:
    if not _has_table(inspector, table_name):
        return False
    return any(constraint["name"] == constraint_name for constraint in inspector.get_check_constraints(table_name))


def _normalize_transport(raw_value: str | None) -> str:
    value = (raw_value or "").strip().lower()
    if value in {"sse", "stdio"}:
        return value
    return "sse"


def _normalize_args(raw_value) -> list[str] | None:
    if raw_value in (None, "", []):
        return None
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value if str(item).strip()]
    if isinstance(raw_value, tuple):
        return [str(item) for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if not candidate:
            return None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return [candidate]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        return None
    return None


def _build_legacy_mcp_row(record) -> dict:
    legacy_payload = {}
    raw_json = record.mcp_setting_json
    if raw_json:
        try:
            loaded = json.loads(raw_json)
            if isinstance(loaded, dict):
                legacy_payload = loaded
        except json.JSONDecodeError:
            legacy_payload = {}

    transport = _normalize_transport(legacy_payload.get("transport"))
    fallback_name = f"legacy-mcp-{record.id}"
    name = str(legacy_payload.get("name") or fallback_name).strip() or fallback_name
    url = legacy_payload.get("url")
    command = legacy_payload.get("command")
    args = _normalize_args(legacy_payload.get("args"))

    is_active = bool(legacy_payload) and (
        bool(url and transport == "sse")
        or bool(command and transport == "stdio")
    )

    if transport == "sse":
        command = None
        args = None
    else:
        url = None

    return {
        "id": record.id,
        "name": name,
        "transport": transport,
        "url": url.strip() if isinstance(url, str) and url.strip() else None,
        "command": command.strip() if isinstance(command, str) and command.strip() else None,
        "args": args,
        "is_active": is_active,
    }


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, 't_user_skill'):
        op.create_table(
            't_user_skill',
            sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column('user_id', sa.BigInteger(), nullable=False),
            sa.Column('name', sa.String(length=100), nullable=False),
            sa.Column('description', sa.String(length=500), nullable=False),
            sa.Column('system_prompt', sa.Text(), nullable=False),
            sa.Column('bound_tools', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
            sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
            sa.Column('create_time', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
            sa.Column('update_time', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
            sa.PrimaryKeyConstraint('id'),
        )
        inspector = sa.inspect(bind)

    if not _has_index(inspector, 't_user_skill', op.f('ix_t_user_skill_user_id')):
        op.create_index(op.f('ix_t_user_skill_user_id'), 't_user_skill', ['user_id'], unique=False)

    if not _has_column(inspector, 't_user_mcp', 'name'):
        op.add_column('t_user_mcp', sa.Column('name', sa.String(length=100), nullable=True))
    if not _has_column(inspector, 't_user_mcp', 'transport'):
        op.add_column('t_user_mcp', sa.Column('transport', sa.String(length=20), server_default=sa.text("'sse'"), nullable=False))
    if not _has_column(inspector, 't_user_mcp', 'url'):
        op.add_column('t_user_mcp', sa.Column('url', sa.String(length=500), nullable=True))
    if not _has_column(inspector, 't_user_mcp', 'command'):
        op.add_column('t_user_mcp', sa.Column('command', sa.String(length=500), nullable=True))
    if not _has_column(inspector, 't_user_mcp', 'args'):
        op.add_column('t_user_mcp', sa.Column('args', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    if not _has_column(inspector, 't_user_mcp', 'is_active'):
        op.add_column('t_user_mcp', sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False))

    inspector = sa.inspect(bind)
    if not _has_check_constraint(inspector, 't_user_mcp', 'ck_t_user_mcp_transport'):
        op.create_check_constraint('ck_t_user_mcp_transport', 't_user_mcp', "transport IN ('sse', 'stdio')")

    op.alter_column('t_user_mcp', 'create_time',
               existing_type=postgresql.TIMESTAMP(),
               nullable=False,
               existing_server_default=sa.text('now()'))
    op.alter_column('t_user_mcp', 'update_time',
               existing_type=postgresql.TIMESTAMP(),
               nullable=False,
               existing_server_default=sa.text('now()'))

    if not _has_index(inspector, 't_user_mcp', op.f('ix_t_user_mcp_user_id')):
        op.create_index(op.f('ix_t_user_mcp_user_id'), 't_user_mcp', ['user_id'], unique=False)

    legacy_rows = bind.execute(
        sa.text("SELECT id, mcp_setting_json FROM t_user_mcp")
    ).fetchall()
    for record in legacy_rows:
        payload = _build_legacy_mcp_row(record)
        bind.execute(
            sa.text(
                """
                UPDATE t_user_mcp
                SET
                    name = :name,
                    transport = :transport,
                    url = :url,
                    command = :command,
                    args = CAST(:args AS JSONB),
                    is_active = :is_active
                WHERE id = :id
                """
            ),
            {
                "id": payload["id"],
                "name": payload["name"],
                "transport": payload["transport"],
                "url": payload["url"],
                "command": payload["command"],
                "args": json.dumps(payload["args"], ensure_ascii=False) if payload["args"] is not None else None,
                "is_active": payload["is_active"],
            },
        )

    op.alter_column('t_user_mcp', 'name', existing_type=sa.String(length=100), nullable=False)
    if _has_column(sa.inspect(bind), 't_user_mcp', 'mcp_setting_json'):
        op.drop_column('t_user_mcp', 'mcp_setting_json')


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_column(inspector, 't_user_mcp', 'mcp_setting_json'):
        op.add_column('t_user_mcp', sa.Column('mcp_setting_json', sa.VARCHAR(length=255), autoincrement=False, nullable=True))
    bind.execute(
        sa.text(
            """
            UPDATE t_user_mcp
            SET mcp_setting_json = jsonb_build_object(
                'name', name,
                'transport', transport,
                'url', url,
                'command', command,
                'args', COALESCE(args, '[]'::jsonb)
            )::text
            """
        )
    )

    inspector = sa.inspect(bind)
    if _has_index(inspector, 't_user_mcp', op.f('ix_t_user_mcp_user_id')):
        op.drop_index(op.f('ix_t_user_mcp_user_id'), table_name='t_user_mcp')
    if _has_check_constraint(inspector, 't_user_mcp', 'ck_t_user_mcp_transport'):
        op.drop_constraint('ck_t_user_mcp_transport', 't_user_mcp', type_='check')
    op.alter_column('t_user_mcp', 'mcp_setting_json', existing_type=sa.VARCHAR(length=255), nullable=False)
    op.alter_column('t_user_mcp', 'update_time',
               existing_type=postgresql.TIMESTAMP(),
               nullable=True,
               existing_server_default=sa.text('now()'))
    op.alter_column('t_user_mcp', 'create_time',
               existing_type=postgresql.TIMESTAMP(),
               nullable=True,
               existing_server_default=sa.text('now()'))
    for column_name in ('is_active', 'args', 'command', 'url', 'transport', 'name'):
        if _has_column(sa.inspect(bind), 't_user_mcp', column_name):
            op.drop_column('t_user_mcp', column_name)
    inspector = sa.inspect(bind)
    if _has_index(inspector, 't_user_skill', op.f('ix_t_user_skill_user_id')):
        op.drop_index(op.f('ix_t_user_skill_user_id'), table_name='t_user_skill')
    if _has_table(inspector, 't_user_skill'):
        op.drop_table('t_user_skill')
