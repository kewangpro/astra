from fastapi import APIRouter

from backend.sandbox.manager import sandbox_manager
from backend.schemas.node import NodeStatus

router = APIRouter(prefix="/nodes", tags=["nodes"])


@router.get("", response_model=list[NodeStatus])
async def list_nodes():
    return sandbox_manager.node_status()
