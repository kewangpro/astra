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


async def recover_interrupted_missions() -> list:
    """
    Called once on application startup. For each RUNNING/PAUSED mission:
      - If sandbox is still alive → terminate it and reset to PENDING so the
        loop can restart cleanly from the last saved checkpoint.
      - If sandbox is gone        → reset to PENDING; loop re-launches from checkpoint.

    Returns the list of mission IDs that need their loop restarted.
    """
    from backend.sandbox.manager import sandbox_manager   # late import to avoid circular dep

    recoverable = [
        MissionStatus.RUNNING.value,
        MissionStatus.PAUSED.value,
        MissionStatus.PLANNING.value,
        MissionStatus.EVALUATING.value,
    ]

    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                select(Mission).where(Mission.status.in_(recoverable))
            )
            missions = result.scalars().all()

            if not missions:
                logger.info("State recovery: no interrupted missions found.")
                return []

            reattached_ids = []
            reset_ids = []

            for mission in missions:
                outcome = sandbox_manager.recover(
                    mission.id,
                    mission.subprocess_pid,
                    mission.container_id,
                )
                if outcome == "reattached":
                    # Terminate the still-running subprocess so the loop can
                    # restart it cleanly from the last checkpoint.
                    sandbox_manager.terminate(mission.id)
                    reattached_ids.append(mission.id)
                else:
                    reset_ids.append(mission.id)

            all_ids = reattached_ids + reset_ids
            if all_ids:
                await session.execute(
                    update(Mission)
                    .where(Mission.id.in_(all_ids))
                    .values(
                        status=MissionStatus.PENDING.value,
                        container_id=None,
                        subprocess_pid=None,
                    )
                )

        if reattached_ids:
            logger.info(
                "State recovery: %d mission(s) sandbox terminated + reset to PENDING: %s",
                len(reattached_ids), reattached_ids,
            )
        if reset_ids:
            logger.info(
                "State recovery: %d mission(s) reset to PENDING (sandbox gone): %s",
                len(reset_ids), reset_ids,
            )

        return all_ids
