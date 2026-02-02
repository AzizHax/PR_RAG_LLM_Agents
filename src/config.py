from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data: Path = root / "data"
    kb: Path = root / "kb"
    outputs: Path = root / "outputs" / "runs"

PATHS = Paths()
