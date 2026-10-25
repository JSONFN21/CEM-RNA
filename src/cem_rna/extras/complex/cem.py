# sample some rna, score it, learn from the best, and repeat
import math
import numpy as np
from .interface.config import Config, MODES, make_config
from .support.scorer import Scorer, score_batch
from ...parallel.parallel_scorer import ScoringPool
from ...cem import CEM as SimpleCEM
from .support.tracking import SearchTracking
from .support.results import build_result

# keep these names around for older scripts
SAMPLES = Config.samples
ELITE_FRACTION = Config.elite_fraction
TEMPERATURE = Config.temperature
INITIALIZATION = Config.initialization
BEAM = Config.beam
PATIENCE = Config.patience
MAX_ROUNDS = Config.max_rounds
MAX_REQUESTS = Config.max_requests


# reuse the simple optimizer's setup and sampling, then add the research options
class CEM(SimpleCEM):
    def __init__(self, target, config=None, *, pool=None, on_step=None):
        super().__init__(target, config or Config(), pool=pool, on_step=on_step)
        self.mode = MODES[self.config.mode]
        self.scorer = Scorer(self.target, self.mode.backend, beam=self.config.beam, lazy_threshold=self.config.lazy_threshold, pool=pool, track_umfe=self.config.track_umfe)
        self.vienna = self.scorer if self.mode.backend == 'vienna' else Scorer(self.target, 'vienna', pool=pool)
        self.best_value = -math.inf
        self.best_sequence = ''
        self.tracking = SearchTracking(self.config)

    # let extras use its existing vienna cache and shared scoring pool
    def check_vienna(self, sequences):
        self.vienna.score(sequences)
        return {sequence: (self.vienna.cache[sequence], self.vienna.cache[sequence].probability) for sequence in sequences}

    def score(self, sequences):
        scores = self.request_scores(sequences)
        values = np.asarray([-s.ned if self.mode.objective == 'ned' else s.logp for s in scores])
        for sequence, value in zip(sequences, scores):
            self.tracking.observe_search(sequence, value, self.steps, self.mode.backend)
            self.record_winner(sequence, value)
        if self.tracking.umfe_sequence is not None:
            # once we find a umfe solution, keep working on p and ned
            # no need to keep proving the same success
            self.scorer.track_umfe = False
        winner = int(np.argmax(values))
        if values[winner] > self.best_value:
            self.best_value, self.best_sequence = float(values[winner]), sequences[winner]
        return values

    # keep the advanced elite checks and optional vienna reordering here
    def select_elites(self, values):
        indices = np.argsort(values)[::-1][:self.config.elites]
        sequences = [self.last_sample_sequences[i] for i in indices]
        selected = self.vienna.score(sequences) if self.mode.elite_backend else self.scorer.score(sequences)
        tracker = self.vienna if self.mode.elite_backend else self.scorer
        self.tracking.observe_elites(sequences, tracker, self.steps, track_ned=bool(self.mode.elite_backend) and self.best_ned_sequence is None)
        if self.mode.elite_backend:
            # let linearpartition pick the elites, then let vienna score their weights
            order = np.argsort([s.logp for s in selected])[::-1]
            indices, selected = indices[order], [selected[i] for i in order]
        return indices, selected

    # use the shared 1-ned weights, or the research log-probability weights
    def elite_weights(self, selected):
        if self.mode.weights == 'ned':
            return super().elite_weights(selected)
        logs = np.asarray([s.logp for s in selected])
        weights = np.exp(np.clip((logs - logs.max()) / self.config.temperature, -self.config.log_clip, 0))
        return weights / weights.sum()

    # ned-only modes watch their best ned objective instead of log p
    @property
    def best_objective(self):
        return self.best_value

    # extra tracking changes the notes, not the shared search loop
    def record_round(self, mean_loss):
        self.tracking.record_round(self, mean_loss)

    # keep the detailed research results separate from the simple output
    def build_result(self, stop, started):
        return build_result(self, stop, started)


def optimize(target, alpha=Config.alpha, uniform=Config.uniform, seed=Config.seed, threads=1, *, pool=None, config=None, on_step=None, engine='lazyoutside', tracking='basic', workers=None, **settings):
    config = config or make_config(engine, tracking, alpha=alpha, uniform=uniform, seed=seed, **settings)
    if pool is not None:
        return CEM(target, config, pool=pool, on_step=on_step).run()
    with ScoringPool(threads if workers is None else workers, score_function=score_batch, sequence_index=2) as local_pool:
        return CEM(target, config, pool=local_pool, on_step=on_step).run()


# let several targets share the scoring workers
def optimize_many(targets, *, config=None, workers=1, concurrent_puzzles=None, seeds=None, engine='lazyoutside', tracking='basic', **settings):
    from ...parallel.batch import optimize_targets
    return optimize_targets(targets, config=config or make_config(engine, tracking, **settings), workers=workers, concurrent_puzzles=concurrent_puzzles, seeds=seeds, optimizer=CEM, score_function=score_batch, sequence_index=2)


class CEM500(CEM):
    def __init__(self, target, alpha=Config.alpha, uniform=Config.uniform, seed=Config.seed, threads=1, **settings):
        super().__init__(target, Config(alpha=alpha, uniform=uniform, seed=seed, **settings))
        self._owned_pool = ScoringPool(threads, score_function=score_batch, sequence_index=2)
        self._owned_pool.__enter__()
        self.scorer.pool = self.vienna.pool = self._owned_pool

    def run(self):
        try:
            return super().run()
        finally:
            self._owned_pool.__exit__(None, None, None)
