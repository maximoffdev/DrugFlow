import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Union

try:
    from pytorch_lightning.loggers import Logger
    from pytorch_lightning.utilities import rank_zero_only
except Exception:  # pragma: no cover
    from lightning.pytorch.loggers import Logger  # type: ignore
    from lightning.pytorch.utilities import rank_zero_only  # type: ignore


class PlainFileLogger(Logger):  # pyright: ignore[reportGeneralTypeIssues]
    """Minimal Lightning logger that appends JSONL records to a local file.

    Records include:
      - hyperparameters (`type: hparams`)
      - metrics (`type: metrics`)

    Intended as a drop-in replacement when W&B is disabled.
    """

    def __init__(
        self,
        save_dir: Union[str, os.PathLike],
        filename: str = "logfile",
        name: str = "plainfile",
        version: Optional[str] = None,
        flush: bool = True,
    ) -> None:
        super().__init__()
        self._name = name
        self._version = version or "0"
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._save_dir / filename
        self._flush = flush
        self._fp: Optional[TextIO] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def log_dir(self) -> str:
        return str(self._save_dir)

    def _ensure_open(self) -> None:
        if self._fp is None:
            # Line-buffered text file
            self._fp = open(self._path, "a", encoding="utf-8", buffering=1)

    @rank_zero_only
    def log_hyperparams(self, params: Any) -> None:
        self._ensure_open()
        assert self._fp is not None
        record = {
            "type": "hparams",
            "time": time.time(),
            "params": self._to_jsonable(params),
        }
        self._fp.write(json.dumps(record, sort_keys=True) + "\n")
        if self._flush:
            self._fp.flush()

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self._ensure_open()
        assert self._fp is not None
        record = {
            "type": "metrics",
            "time": time.time(),
            "step": step,
            "metrics": self._to_jsonable(metrics),
        }
        self._fp.write(json.dumps(record, sort_keys=True) + "\n")
        if self._flush:
            self._fp.flush()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._fp is not None:
            record = {"type": "finalize", "time": time.time(), "status": status}
            self._fp.write(json.dumps(record, sort_keys=True) + "\n")
            if self._flush:
                self._fp.flush()
            self._fp.close()
            self._fp = None

    def save(self) -> None:
        # No additional artifacts to save.
        return

    def _to_jsonable(self, obj: Any) -> Any:
        # Keep this conservative: only convert common containers.
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            return self._to_jsonable(obj.__dict__)
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)
