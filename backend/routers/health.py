import platform
import psutil
from fastapi import APIRouter
from backend.config import settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check():
    mem = psutil.virtual_memory()
    return {
        "status": "ok",
        "env": settings.env,
        "autonomy_mode": settings.autonomy_mode,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "total_memory_gb": round(mem.total / (1024**3), 1),
            "available_memory_gb": round(mem.available / (1024**3), 1),
        },
    }


@router.get("/ready")
async def readiness():
    return {"ready": True}
