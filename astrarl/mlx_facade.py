"""MLX learner facade — shared Learner protocol over Ensemble finetune CLIs.

Used by AstraRL (online / gym-coupled) and AstraTune (offline / dataset) via
``FacadeConfig``. ``use_execv=True`` for zero-orphan mission sandboxes.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from astrarl.mlx_learners import ENTRYPOINTS, entrypoint_for, gym_id_for_mlx_learner


@dataclass(frozen=True)
class FacadeConfig:
    """Package-specific wiring for ``MLXLearnerFacade`` (ENTRYPOINTS + flag quirks)."""

    entrypoints: Mapping[str, str]
    entrypoint_for: Callable[[str], str]
    facade_only: frozenset[str]
    flag_aliases: Mapping[str, str] = field(default_factory=dict)
    resolve_gym_id: Optional[Callable[..., str]] = None
    # ``(learner_id, merged_kwargs, total_timesteps) -> None`` mutates merged in place.
    map_total_timesteps: Optional[Callable[[str, dict[str, Any], int], None]] = None
    unknown_learner_fmt: str = "not an MLX learner_id: {learner_id!r}"
    predict_msg_fmt: str = (
        "{learner_id}: inference is served by Ensemble/JEV MLX endpoints, "
        "not MLXLearnerFacade.predict()"
    )


def _astrarl_map_timesteps(learner_id: str, merged: dict[str, Any], total_timesteps: int) -> None:
    if "iters" not in merged:
        merged["iters"] = int(total_timesteps)


_ASTRARL_FACADE_ONLY = frozenset({
    "reward_mode",
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
})


ASTRARL_FACADE_CONFIG = FacadeConfig(
    entrypoints=ENTRYPOINTS,
    entrypoint_for=entrypoint_for,
    facade_only=_ASTRARL_FACADE_ONLY,
    resolve_gym_id=lambda learner_id, **kw: gym_id_for_mlx_learner(
        learner_id, reward_mode=str(kw.get("reward_mode", "") or ""),
    ),
    map_total_timesteps=_astrarl_map_timesteps,
)


def _flagize(key: str, aliases: Mapping[str, str]) -> str:
    if key in aliases:
        return aliases[key]
    return "--" + str(key).replace("_", "-")


def kwargs_to_argv(
    kwargs: dict[str, Any],
    *,
    facade_only: Optional[frozenset[str]] = None,
    flag_aliases: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """Map facade/recipe kwargs to Ensemble CLI argv (booleans → store_true flags)."""
    only = facade_only if facade_only is not None else _ASTRARL_FACADE_ONLY
    aliases = flag_aliases if flag_aliases is not None else {}
    cold = bool(kwargs.get("no_adapter") or kwargs.get("no_warmstart"))
    argv: list[str] = []
    for key, val in kwargs.items():
        if key in only or val is None:
            continue
        if cold and key == "adapter":
            continue
        flag = _flagize(key, aliases)
        if isinstance(val, bool):
            if val:
                argv.append(flag)
            continue
        argv.extend([flag, str(val)])
    return argv


class MLXLearnerFacade:
    """Thin Learner that shells to an Ensemble ``*_train.py`` entrypoint."""

    def __init__(
        self,
        learner_id: str,
        gym: Any = None,
        *,
        config: Optional[FacadeConfig] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or ASTRARL_FACADE_CONFIG
        if learner_id not in cfg.entrypoints:
            raise ValueError(cfg.unknown_learner_fmt.format(learner_id=learner_id))
        self.config = cfg
        self.learner_id = learner_id
        self.gym = gym
        self.kwargs = dict(kwargs)
        self.entrypoint = cfg.entrypoint_for(learner_id)
        if cfg.resolve_gym_id is not None:
            self.gym_id = cfg.resolve_gym_id(learner_id, **kwargs)
        else:
            self.gym_id = None
        self._last_save_dir: Optional[str] = (
            str(kwargs["save_dir"]) if kwargs.get("save_dir") else None
        )

    def descriptor(self) -> dict[str, Any]:
        """Legacy dict shape for callers that only need entrypoint / gym_id."""
        out = {
            "learner_id": self.learner_id,
            "entrypoint": self.entrypoint,
            "gym": self.gym,
            "kwargs": dict(self.kwargs),
        }
        if self.gym_id is not None:
            out["gym_id"] = self.gym_id
        return out

    def build_argv(
        self,
        *,
        total_timesteps: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        """Full argv: [python, script, ...flags]."""
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
        if total_timesteps is not None and self.config.map_total_timesteps is not None:
            self.config.map_total_timesteps(self.learner_id, merged, int(total_timesteps))

        argv = [str(python_bin), script]
        argv.extend(
            kwargs_to_argv(
                merged,
                facade_only=self.config.facade_only,
                flag_aliases=self.config.flag_aliases,
            )
        )
        extra_argv = merged.get("extra_argv") or self.kwargs.get("extra_argv")
        if extra_argv:
            argv.extend(list(extra_argv))
        return argv

    def learn(self, total_timesteps: int = 0, **kwargs: Any) -> dict[str, Any]:
        """Run the Ensemble entrypoint (subprocess by default; optional execv)."""
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
            self.config.predict_msg_fmt.format(learner_id=self.learner_id)
        )

    def save(self, path: str) -> str:
        """Record the adapter/save directory; Ensemble CLI writes checkpoints."""
        self._last_save_dir = path
        self.kwargs["save_dir"] = path
        return path

    def load(self, path: str) -> None:
        """Point warm-start / adapter path at ``path`` for the next ``learn()``."""
        self.kwargs["adapter"] = path
        self._last_save_dir = path


def make_mlx_facade(learner_id: str, gym: Any = None, **kwargs: Any) -> MLXLearnerFacade:
    return MLXLearnerFacade(learner_id, gym, config=ASTRARL_FACADE_CONFIG, **kwargs)
