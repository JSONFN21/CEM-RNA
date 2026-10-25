# settings and choices for scoring and tracking
from dataclasses import dataclass
import math
from ....config import Config as SimpleConfig


@dataclass(frozen=True)
class Mode:
    backend: str
    objective: str
    weights: str
    elite_backend: str | None = None


MODES = {
    'vienna-ned': Mode('vienna', 'ned', 'ned'),
    'vienna-1ned': Mode('vienna', 'logp', 'ned'),
    'vienna-log': Mode('vienna-log', 'logp', 'logp'),
    'lazy-ned': Mode('lazy', 'ned', 'ned'),
    'lazy-1ned': Mode('lazy', 'logp', 'ned'),
    'lp-log': Mode('lp', 'logp', 'logp'),
    'lp-vienna-1ned': Mode('lp', 'logp', 'ned', 'vienna'),
    'lp-vienna-log': Mode('lp', 'logp', 'logp', 'vienna'),
}
DEFAULT_MODE = 'lazy-1ned'


@dataclass(frozen=True)
class Config(SimpleConfig):
    mode: str = DEFAULT_MODE
    temperature: float = 1.0
    track_sep: bool = False
    track_umfe: bool = False
    track_all: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in MODES:
            raise ValueError(f'unknown mode: {self.mode}')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('temperature must be greater than 0')


# these choices all rank by p and learn with 1-ned weights
ENGINES = {'lazyoutside': 'lazy-1ned', 'hybrid': 'lp-vienna-1ned', 'vienna': 'vienna-1ned'}
TRACKING = {
    'basic': {},
    'elites': {'track_all': True},
    'umfe': {'track_umfe': True, 'track_sep': True},
}


# turn the friendly options into search settings
def make_config(engine='lazyoutside', tracking='basic', **settings):
    if engine not in ENGINES:
        raise ValueError(f'unknown engine: {engine}')
    if tracking not in TRACKING:
        raise ValueError(f'unknown tracking option: {tracking}')
    options = dict(mode=ENGINES[engine], **TRACKING[tracking])
    options.update(settings)
    return Config(**options)
