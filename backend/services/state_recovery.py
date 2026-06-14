"""
Boot-time state recovery: finds RUNNING/PAUSED missions and resets them to PENDING
using atomic DB transactions. Sandbox re-attachment is deferred to Phase 2 (Step 2.1).
"""
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.mission import Mission, MissionStatus
from backend.database import AsyncSessionLocal
from backend.logging_config import get_logger

logger = get_logger(__name__)


async def recover_interrupted_missions() -> int:
    """
    Called once on application startup. Atomically resets any missions that were
    left in RUNNING or PAUSED state — they cannot be reliably resumed without
    sandbox re-attachment (Phase 2), so we reset to PENDING.

    Returns the number of missions that were recovered.
    """
    recoverable = [MissionStatus.RUNNING, MissionStatus.PAUSED]

    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                select(Mission).where(Mission.status.in_([s.value for s in recoverable]))
            )
            missions = result.scalars().all()

            if not missions:
                logger.info("State recovery: no interrupted missions found.")
                return 0

            ids = [m.id for m in missions]
            await session.execute(
                update(Mission)
                .where(Mission.id.in_(ids))
                .values(
                    status=MissionStatus.PENDING.value,
                    # Clear stale sandbox references — re-attachment handled in Phase 2
                    container_id=None,
                    subprocess_pid=None,
                )
            )

        logger.warning(
            "State recovery: reset %d mission(s) to PENDING: %s",
            len(ids),
            ids,
        )
        return len(ids)
