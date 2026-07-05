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


async def recover_interrupted_missions() -> dict:
    """
    Called once on application startup. For each RUNNING/PAUSED mission:
      - Sandbox still running → reattach and resume polling it in place;
        mission keeps its status/remote_pid/subprocess_pid untouched.
      - Sandbox gone → reset to PENDING; loop re-launches from checkpoint.

    Returns {"restart": [...], "resume": [...]} — mission IDs needing a fresh
    `loop.run()` vs a `loop.run(resume_existing_sandbox=True)` reattach.
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
                return {"restart": [], "resume": []}

            resume_ids = []   # still alive — reattach in place, don't touch
            reset_ids = []    # sandbox gone — restart fresh

            for mission in missions:
                outcome = sandbox_manager.recover(
                    mission.id,
                    mission.subprocess_pid,
                    mission.container_id,
                    remote_pid=mission.remote_pid,
                )
                if outcome == "reattached":
                    resume_ids.append(mission.id)
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
                        remote_pid=None,
                    )
                )

        if resume_ids:
            logger.info(
                "State recovery: %d mission(s) reattached and resuming in place: %s",
                len(resume_ids), resume_ids,
            )
        if reset_ids:
            logger.info(
                "State recovery: %d mission(s) reset to PENDING (sandbox gone): %s",
                len(reset_ids), reset_ids,
            )

        return {"restart": reset_ids, "resume": resume_ids}
