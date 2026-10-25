# get p and ned together, and remember scores we already know
from dataclasses import dataclass
from importlib import import_module
import time
from .config import Config


@dataclass(frozen=True)
class Score:
    logp: float
    ned: float


def native_module():
    return import_module('cem_rna._lazyoutside_native')


def score_batch(target, sequences, beam, lazy_threshold):
    started = time.process_time()
    return lazy_scores(target, sequences, beam, lazy_threshold), time.process_time() - started


# both optimizers use the same lazyoutside scoring pass
def lazy_scores(target, sequences, beam, lazy_threshold, score_type=Score):
    values = native_module().score_batch(sequences, target, beam, lazy_threshold, True)
    return [score_type(float(logp), float(ned)) for logp, ned in values]


class Scorer:
    def __init__(self, target, *, beam=Config.beam, lazy_threshold=Config.lazy_threshold, pool=None):
        self.target = target
        self.beam, self.lazy_threshold, self.pool = beam, lazy_threshold, pool
        self.cache = {}
        self.cpu_seconds = 0.0

    # leave generated sequences as they are, while extras also accepts dna and lowercase
    @staticmethod
    def normalize(sequences):
        return sequences

    # run a batch here or in the pool, and add up its cpu time
    def run_batch(self, function, arguments):
        values, cpu = function(*arguments) if self.pool is None else self.pool.score(*arguments)
        self.cpu_seconds += cpu
        return values

    # score new sequences with the simple lazyoutside backend
    def evaluate(self, sequences):
        arguments = (self.target, sequences, self.beam, self.lazy_threshold)
        return self.run_batch(score_batch, arguments)

    # score each new sequence once, in the order we first saw it
    def score_missing(self, sequences):
        missing = list(dict.fromkeys(sequence for sequence in sequences if sequence not in self.cache))
        if missing:
            self.cache.update(zip(missing, self.evaluate(missing)))

    # return scores in the original order, repeats and all
    def score(self, sequences):
        normalized = self.normalize(sequences)
        self.score_missing(normalized)
        return [self.cache[sequence] for sequence in normalized]
