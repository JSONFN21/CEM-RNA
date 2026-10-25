import math
from numbers import Integral


def check_integer(name, value, minimum=1):
    if not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer of at least {minimum}')


# check settings once before starting the search
def check_config(config):
    for name in ('samples', 'beam', 'patience', 'max_rounds'):
        check_integer(name, getattr(config, name))
    check_integer('seed', config.seed, 0)
    if config.max_requests is not None:
        check_integer('max_requests', config.max_requests)
    for name in ('alpha', 'uniform', 'initialization'):
        value = getattr(config, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be between 0 and 1')
    for name in ('elite_fraction', 'probability_floor'):
        value = getattr(config, name)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'{name} must be greater than 0 and at most 1')
    if not math.isfinite(config.lazy_threshold) or not 0 < config.lazy_threshold < 1:
        raise ValueError('lazy_threshold must be greater than 0 and less than 1')
    if not math.isfinite(config.log_clip) or config.log_clip <= 0:
        raise ValueError('log_clip must be greater than 0')


def check_workers(workers, concurrent_puzzles=None, seeds=None):
    check_integer('workers', workers)
    if concurrent_puzzles is not None:
        check_integer('concurrent_puzzles', concurrent_puzzles)
    if seeds is not None:
        for seed in seeds:
            check_integer('seed', seed, 0)
