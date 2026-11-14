"""MLX offline learner facade — AstraTune platform over Ensemble finetune CLIs.

Mirrors ``astrarl.mlx_facade`` for dataset / preference workers (SFT, DPO,
distill, RFT, STAR, RLCD). ``use_execv=True`` for zero-orphan mission sandboxes.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Optional

from astratune.mlx_learners import ENTRYPOINTS, entrypoint_for


_FACADE_ONLY = frozenset({
    "finetune_dir",
    "python_bin",
    "extra_argv",
    "use_execv",
    "env",
    "cwd",
    "check",
    "gym_id",
    "learner_id",
    "task_type",
    "algorithm",
    "gym",
})

# argparse names that are not simple ``--{key.replace('_','-')}``.
_FLAG_ALIASES: dict[str, str] = {
    "k_collect": "--K-collect",
    "K_collect": "--K-collect",
}


def _flagize(key: str) -> str:
    if key in _FLAG_ALIASES:
        return _FLAG_ALIASES[key]
    return "--" + str(key).replace("_", "-")


def kwargs_to_argv(kwargs: dict[str, Any]) -> list[str]:
    """Map facade/recipe kwargs to Ensemble CLI argv (booleans → store_true flags)."""
    argv: list[str] = []
    cold = bool(kwargs.get("no_adapter") or kwargs.get("no_warmstart"))
    for key, val in kwargs.items():
        if key in _FACADE_ONLY or val is None:
            continue
        if cold and key == "adapter":
            continue
        flag = _flagize(key)
        if isinstance(val, bool):
            if val:
                argv.append(flag)
            continue
        argv.extend([flag, str(val)])
    return argv


class MLXLearnerFacade:
    """Thin Learner that shells to Ensemble ``dpo_train.py`` / SFT peers."""

    def __init__(
        self,
        learner_id: str,
        gym: Any = None,
        **kwargs: Any,
    ) -> None:
        if learner_id not in ENTRYPOINTS:
            raise ValueError(f"not an AstraTune MLX learner_id: {learner_id!r}")
        self.learner_id = learner_id
        self.gym = gym
        self.kwargs = dict(kwargs)
        self.entrypoint = entrypoint_for(learner_id)
        self._last_save_dir: Optional[str] = (
            str(kwargs["save_dir"]) if kwargs.get("save_dir") else None
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "entrypoint": self.entrypoint,
            "gym": self.gym,
            "kwargs": dict(self.kwargs),
        }

    def build_argv(
        self,
        *,
        total_timesteps: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        finetune_dir = self.kwargs.get("finetune_dir") or os.environ.get(
            "ASTRA_FINETUNE_DIR", "",
        )
        python_bin = self.kwargs.get("python_bin") or sys.executable
        script = self.entrypoint
        if finetune_dir:
            script = os.path.join(str(finetune_dir), self.entrypoint)

        merged = dict(self.kwargs)
        if extra:
            merged.update(extra)
        if total_timesteps is not None:
            if "iters" not in merged and self.learner_id != "mlx_dpo":
                merged["iters"] = int(total_timesteps)
            if "epochs" not in merged and self.learner_id == "mlx_dpo":
                merged["epochs"] = int(total_timesteps)

        argv = [str(python_bin), script]
        argv.extend(kwargs_to_argv(merged))
        extra_argv = merged.get("extra_argv") or self.kwargs.get("extra_argv")
        if extra_argv:
            argv.extend(list(extra_argv))
        return argv

    def learn(self, total_timesteps: int = 0, **kwargs: Any) -> dict[str, Any]:
        argv = self.build_argv(
            total_timesteps=total_timesteps if total_timesteps else None,
            extra=kwargs or None,
        )
        use_execv = bool(self.kwargs.get("use_execv", False))
        cwd = self.kwargs.get("cwd") or self.kwargs.get("finetune_dir") or None
        env = os.environ.copy()
        if isinstance(self.kwargs.get("env"), dict):
            env.update({str(k): str(v) for k, v in self.kwargs["env"].items()})

        if use_execv:
            if cwd:
                os.chdir(str(cwd))
            os.execv(argv[0], argv)

        check = self.kwargs.get("check", True)
        proc = subprocess.run(argv, cwd=cwd, env=env, check=False)
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"{self.entrypoint} exited {proc.returncode} (argv={argv!r})"
            )
        save_dir = kwargs.get("save_dir") or self.kwargs.get("save_dir")
        if save_dir:
            self._last_save_dir = str(save_dir)
        return {
            "learner_id": self.learner_id,
            "entrypoint": self.entrypoint,
            "returncode": proc.returncode,
            "argv": argv,
            "total_timesteps": int(total_timesteps) if total_timesteps else None,
        }

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        raise NotImplementedError(
            f"{self.learner_id}: inference is served by Ensemble/JEV MLX endpoints, "
            "not astratune.MLXLearnerFacade.predict()"
        )

    def save(self, path: str) -> str:
        self._last_save_dir = path
        self.kwargs["save_dir"] = path
        return path

    def load(self, path: str) -> None:
        self.kwargs["adapter"] = path
        self._last_save_dir = path


def make_mlx_facade(learner_id: str, gym: Any = None, **kwargs: Any) -> MLXLearnerFacade:
    return MLXLearnerFacade(learner_id, gym, **kwargs)
