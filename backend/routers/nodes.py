from fastapi import APIRouter
from sqlalchemy import select

from backend.database import AsyncSessionLocal
from backend.models.mission import Mission, MissionStatus
from backend.sandbox.manager import sandbox_manager
from backend.schemas.node import NodeStatus

router = APIRouter(prefix="/nodes", tags=["nodes"])


@router.get("", response_model=list[NodeStatus])
async def list_nodes():
    nodes = sandbox_manager.node_status()
    # Ensure consistency between top KPI bar and Nodes panel:
    # Any active mission in the DB (running, planning, evaluating, pending)
    # is tracked on its assigned host even if its sandbox subprocess has not yet spawned.
    async with AsyncSessionLocal() as session:
        stmt = select(Mission).where(
            Mission.status.in_([
                MissionStatus.RUNNING.value,
                MissionStatus.PLANNING.value,
                MissionStatus.EVALUATING.value,
                MissionStatus.PENDING.value,
            ])
        )
        result = await session.execute(stmt)
        active_missions = result.scalars().all()

    if active_missions:
        nodes_by_host = {n["host"]: n for n in nodes}
        for m in active_missions:
            host = m.host or "local"
            if host in nodes_by_host:
                existing_mids = {x["mission_id"] for x in nodes_by_host[host]["missions"]}
                if m.id not in existing_mids:
                    nodes_by_host[host]["missions"].append({
                        "mission_id": m.id,
                        "sandbox_id": str(m.subprocess_pid) if m.subprocess_pid else (str(m.remote_pid) if m.remote_pid else None),
                    })
                    if host == "local":
                        nodes_by_host[host]["alive"] = True

    return nodes
