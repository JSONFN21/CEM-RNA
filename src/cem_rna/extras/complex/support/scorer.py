# remember sequence scores and structure checks for the scoring options
from dataclasses import dataclass, replace
import math
import time
import numpy as np
from ..interface.config import Config
from ....scorer import Scorer as SimpleScorer, Score as SimpleScore, native_module, lazy_scores
from ....validation import fold_sequence, mfe_status

CENTIKCAL_PER_KCAL = 100


# scores for one sequence, plus any exact structure checks
@dataclass(frozen=True)
class Score(SimpleScore):

    ned: float | None = None
    probability: float | None = None
    is_mfe: bool | None = None
    is_umfe: bool | None = None
    mfe_cpu_seconds: float = 0.0
    metrics: dict | None = None


# ask vienna for exact p and, when needed, ned
def vienna_scores(sequences, target, *, compute_ned=True, track_umfe=False):
    values = []
    for sequence in sequences:
        exact, probability, fold, structure, mfe = fold_sequence(sequence, target, compute_ned=compute_ned)
        value = Score(exact.logp, exact.ned, probability)
        if track_umfe:
            value = add_structure_metrics(sequence, target, value, fold, structure, mfe)
        values.append(value)
    return values


# score one batch here or in the shared worker pool
def score_batch(backend, target, sequences, beam, lazy_threshold, track_umfe=False):
    started = time.process_time()
    if backend == 'structure':
        values = [add_structure_metrics(sequence, target, value) for sequence, value in sequences]
        return values, time.process_time() - started
    if backend in ('vienna', 'vienna-log'):
        values = vienna_scores(sequences, target, compute_ned=backend == 'vienna', track_umfe=track_umfe)
    elif backend == 'lazy':
        values = lazy_scores(target, sequences, beam, lazy_threshold, score_type=Score)
    elif backend == 'lp':
        log_probabilities = native_module().score_log_batch(sequences, target, beam)
        values = [Score(logp=float(logp)) for logp in log_probabilities]
    else:
        raise ValueError(f'Unknown backend: {backend}')
    if track_umfe and backend not in ('vienna', 'vienna-log'):
        values = [add_structure_metrics(sequence, target, value) for sequence, value in zip(sequences, values)]
    return values, time.process_time() - started


# remember scores for one target and send new work to the pool
class Scorer(SimpleScorer):
    def __init__(self, target, backend, *, beam=Config.beam, lazy_threshold=Config.lazy_threshold, pool=None, track_umfe=False):
        super().__init__(target.strip(), beam=beam, lazy_threshold=lazy_threshold, pool=pool)
        self.backend = backend
        self.track_umfe = track_umfe

    @staticmethod
    def normalize(sequences):
        return [s.strip().upper().replace('T', 'U') for s in sequences]

    # choose the research backend and reuse the shared cache and timing
    def evaluate(self, sequences):
        return self.evaluate_backend(self.backend, sequences, self.track_umfe)

    # structure checks use this same path without replacing the search backend
    def evaluate_backend(self, backend, sequences, track_umfe=False):
        arguments = (backend, self.target, sequences, self.beam, self.lazy_threshold, track_umfe)
        return self.run_batch(score_batch, arguments)

    # add vienna structure checks without recalculating cached p and ned
    def score_structures(self, sequences, observations=None):
        normalized = self.normalize(sequences)
        self.score_missing(normalized)
        if observations is not None:
            self._reuse_structure_metrics(normalized, observations)

        missing = list(dict.fromkeys(sequence for sequence in normalized if self.cache[sequence].metrics is None))
        if missing:
            candidates = [(sequence, self.cache[sequence]) for sequence in missing]
            values = self.evaluate_backend('structure', candidates)
            self.cache.update(zip(missing, values))
        return [self.cache[sequence] for sequence in normalized]

    # reuse the structure details but keep this scorer's p and ned
    def _reuse_structure_metrics(self, sequences, observations):
        for sequence in dict.fromkeys(sequences):
            cached = self.cache[sequence]
            observed = observations.get(sequence)
            if cached.metrics is not None or observed is None or observed.metrics is None:
                continue

            metrics = dict(
                observed.metrics,
                probability=cached.probability,
                log_probability=cached.logp,
                normalized_ensemble_defect=cached.ned,
            )
            self.cache[sequence] = replace(cached, metrics=metrics, is_mfe=observed.is_mfe, is_umfe=observed.is_umfe)

    def score_log_many(self, sequences):
        return np.asarray([value.logp for value in self.score(sequences)])


# keep these handy scorer names for older notebooks
class LinearPartitionScorer(Scorer):
    def __init__(self, target, beam=Config.beam):
        super().__init__(target, 'lp', beam=beam)


class ViennaRNAScorer(Scorer):
    def __init__(self, target):
        super().__init__(target, 'vienna')


# get exact structure details and reuse the p and ned we already have
def structure_metrics(sequence, target, score, fold=None, structure=None, mfe=None):
    import RNA

    if fold is None:
        fold = RNA.fold_compound(sequence)
    if structure is None:
        structure, mfe = fold.mfe()
    energy = float(fold.eval_structure(target))
    gap = (round(energy * CENTIKCAL_PER_KCAL) - round(mfe * CENTIKCAL_PER_KCAL)) / CENTIKCAL_PER_KCAL
    target_pairs = RNA.ptable(target)
    mfe_pairs = RNA.ptable(structure)
    distance = sum(target_pairs[i] != mfe_pairs[i] for i in range(1, len(target) + 1))
    is_mfe, is_umfe = mfe_status(sequence, target, fold, structure, mfe)
    return {
        'sequence': sequence,
        'probability': score.probability,
        'log_probability': score.logp,
        'normalized_ensemble_defect': score.ned,
        'mfe_structure': structure,
        'target_free_energy_kcal_mol': energy,
        'mfe_free_energy_kcal_mol': float(mfe),
        'energy_gap_kcal_mol': gap,
        'mfe_structure_distance': distance,
        'is_mfe': is_mfe,
        'is_umfe': is_umfe,
    }


# add structure details once, using an existing vienna fold when we can
def add_structure_metrics(sequence, target, value, fold=None, structure=None, mfe=None):
    started = time.process_time()
    probability = value.probability if value.probability is not None else math.exp(value.logp)
    metrics = structure_metrics(sequence, target, replace(value, probability=probability), fold, structure, mfe)
    return replace(
        value,
        metrics=metrics,
        is_mfe=metrics['is_mfe'],
        is_umfe=metrics['is_umfe'],
        mfe_cpu_seconds=time.process_time() - started,
    )
