# settings for ranking by p and learning with 1-ned weights
from dataclasses import dataclass
import math
from .checks import check_config


@dataclass(frozen=True)
class Config:
    samples: int = 500
    elite_fraction: float = 0.1
    alpha: float = 0.5
    uniform: float = 0.002
    seed: int = 706
    initialization: float = 0.75
    beam: int = 250
    lazy_threshold: float = 0.00005
    patience: int = 25
    max_rounds: int = 2000
    max_requests: int | None = 5_000_000
    # keep tiny probabilities and exponentials well behaved, this might not be needed?
    probability_floor: float = 1e-15
    log_clip: float = 60.0
    verbose: bool = False

    def __post_init__(self):
        check_config(self)

    @property
    def elites(self):
        return math.ceil(self.samples * self.elite_fraction)
