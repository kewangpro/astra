"""
Boot-time state recovery.

Step 1.2: atomically resets RUNNING/PAUSED missions to PENDING.
Step 2.1: extended with sandbox re-attachment — keeps missions RUNNING if their
          sandbox is still alive; resets to PENDING only when the sandbox is gone.
"""
from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.mission import Mission, MissionStatus
from backend.database import AsyncSessionLocal
from backend.logging_config import get_logger

logger = get_logger(__name__)


async def recover_interrupted_missions() -> int:
    """
    Called once on application startup. For each RUNNING/PAUSED mission:
      - If sandbox is still alive → keep RUNNING, re-attach telemetry back-fill.
      - If sandbox is gone        → reset to PENDING; caller re-launches from checkpoint.

    Returns the number of missions that were acted upon.
    """
    from backend.sandbox.manager import sandbox_manager   # late import to avoid circular dep

    recoverable = [MissionStatus.RUNNING.value, MissionStatus.PAUSED.value]

    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                select(Mission).where(Mission.status.in_(recoverable))
            )
            missions = result.scalars().all()

            if not missions:
                logger.info("State recovery: no interrupted missions found.")
                return 0

            reattached_ids = []
            reset_ids = []

            for mission in missions:
                outcome = sandbox_manager.recover(
                    mission.id,
                    mission.subprocess_pid,
                    mission.container_id,
                )
                if outcome == "reattached":
                    reattached_ids.append(mission.id)
                else:
                    reset_ids.append(mission.id)

            if reset_ids:
                await session.execute(
                    update(Mission)
                    .where(Mission.id.in_(reset_ids))
                    .values(
                        status=MissionStatus.PENDING.value,
                        container_id=None,
                        subprocess_pid=None,
                    )
                )

        if reattached_ids:
            logger.info(
                "State recovery: %d mission(s) reattached (sandbox still alive): %s",
                len(reattached_ids), reattached_ids,
            )
        if reset_ids:
            logger.warning(
                "State recovery: %d mission(s) reset to PENDING (sandbox gone): %s",
                len(reset_ids), reset_ids,
            )

        return len(missions)
