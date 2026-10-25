# this file defines the simple cem optimizer, see extras/complex for a more complicated version with more tracking and options
from dataclasses import asdict
import math
import time
import numpy as np
from .config import Config
from .scorer import Scorer, native_module
from .validation import validate_finalists, validate_structure
from .parallel.parallel_scorer import ScoringPool
from .structure_parser import build_coupled_design_variables


# stop before the next batch goes over our scoring budget
class RequestLimit(Exception):
    pass


class CEM:
    # set up the target, settings, and score cache
    def __init__(self, target, config=None, *, pool=None, on_step=None):
        self.config = config or Config()
        self.target = target.strip()
        self.variables, _, _ = build_coupled_design_variables(self.target)
        self.rng = np.random.default_rng(self.config.seed)
        self.scorer = Scorer(self.target, beam=self.config.beam, lazy_threshold=self.config.lazy_threshold, pool=pool)
        self.on_step = on_step

        # set up our base choices and the fast sequence builder
        self.logits = self.initialize()
        self._probability_logits = self._probability_values = None
        self._positions = [list(v.positions) for v in self.variables]
        self._states = [list(v.states) for v in self.variables]
        self.converter = native_module().assignments_to_sequences

        # keep a separate winner for p and ned
        self.requests = self.steps = 0
        self.best_log = -math.inf
        self.best_log_sequence = ''
        self.best_ned = math.inf
        self.best_ned_sequence = None
        self.vienna_scores = {}
        self.vienna_cpu_seconds = 0.0
        self.best_vienna_probability = 0.0
        self.best_vienna_ned = math.inf
        self.history = []
        self.best_stale_steps = self.objective_stale_steps = 0
        self.best_found_step = None

    # favor a at unpaired spots and cg or gc at paired spots
    def initialize(self):
        bias = self.config.initialization
        base = {b: (1 - bias) / 4 for b in 'ACGU'}
        base['A'] += bias
        logits = []
        for variable in self.variables:
            if variable.kind == 'unpaired':
                p = np.full(4, (1 - bias) / 4)
                p[0] += bias
            elif variable.kind == 'paired':
                p = np.full(6, (1 - bias) / 6)
                p[2:4] += bias / 2
            else:
                p = np.array([math.prod(base[b] for b in state) for state in variable.states])
                p /= p.sum()
            logits.append(np.log(np.clip(p, self.config.probability_floor, 1)))
        return logits

    def probabilities(self):
        if self._probability_logits is not self.logits:
            result = []
            for logits in self.logits:
                p = np.exp(np.clip(logits - logits.max(), -self.config.log_clip, self.config.log_clip))
                result.append(p / p.sum())
            self._probability_logits, self._probability_values = self.logits, result
        return self._probability_values

    def sequences(self, assignments):
        return self.converter(np.asarray(assignments, dtype=np.int16), self._positions, self._states, len(self.target))

    def sequence(self, assignment):
        return self.sequences([assignment])[0]

    def argmax(self):
        return self.sequence([np.argmax(x) for x in self.logits])

    # add a little uniform when sampling, without changing what we learned
    def sample(self):
        uniforms = self.rng.random((self.config.samples, len(self.variables)))
        assignments = np.zeros(uniforms.shape, dtype=np.int16)
        for i, p in enumerate(self.probabilities()):
            mixed = (1 - self.config.uniform) * p + self.config.uniform / len(p)
            states = np.searchsorted(np.cumsum(mixed), uniforms[:, i], side='right')
            assignments[:, i] = np.minimum(states, len(p) - 1)

        sequences = self.sequences(assignments)
        self.last_sample_sequences = sequences
        return assignments, sequences

    # count requests before sending the batch to the scorer
    def request_scores(self, sequences):
        if self.config.max_requests is not None and self.requests + len(sequences) > self.config.max_requests:
            raise RequestLimit
        self.requests += len(sequences)
        return self.scorer.score(sequences)

    # keep separate p and ned winners, without replacing ties
    def record_winner(self, sequence, value):
        if value.logp > self.best_log:
            self.best_log, self.best_log_sequence = value.logp, sequence
        if value.ned is not None and value.ned < self.best_ned:
            self.best_ned, self.best_ned_sequence = value.ned, sequence

    def score(self, sequences):
        scores = self.request_scores(sequences)
        for sequence, value in zip(sequences, scores):
            self.record_winner(sequence, value)
        return np.asarray([s.logp for s in scores])

    # reuse exact checks when the same argmax comes back
    def check_vienna(self, sequences):
        missing = list(dict.fromkeys(s for s in sequences if s not in self.vienna_scores))
        if missing:
            scores, cpu = validate_finalists(self.target, missing)
            self.vienna_scores.update(scores)
            self.vienna_cpu_seconds += cpu
        return self.vienna_scores

    # keep the search scores, then check the most likely sequence with vienna
    def score_argmax(self):
        sequence = self.argmax()
        self.score([sequence])
        scores = self.check_vienna([sequence])
        score, probability = scores[sequence]
        self.best_vienna_probability = max(self.best_vienna_probability, probability)
        self.best_vienna_ned = min(self.best_vienna_ned, score.ned)

    # pick the top samples and reuse their cached scores
    def select_elites(self, values):
        indices = np.argsort(values)[::-1][:self.config.elites]
        sequences = [self.last_sample_sequences[i] for i in indices]

        # we already scored these samples, so reuse the answers
        selected = self.scorer.score(sequences)
        return indices, selected

    # turn 1-ned into weights that add up to one
    def elite_weights(self, selected):
        raw = 1 - np.asarray([s.ned for s in selected])
        return raw / raw.sum() if raw.sum() > 0 else np.full(len(raw), 1 / len(raw))

    # learn the base choices from the elites
    def update(self, assignments, values):
        indices, selected = self.select_elites(values)
        weights = self.elite_weights(selected)
        elite = [np.bincount(assignments[indices, i], weights=weights, minlength=variable.n_states).astype(float) for i, variable in enumerate(self.variables)]
        elite = [p / p.sum() for p in elite]

        # let alpha decide how much we learn, including zero and one
        with np.errstate(divide='ignore'):
            self.logits = [np.log((1 - self.config.alpha) * old + self.config.alpha * new) for old, new in zip(self.probabilities(), elite)]

    # the simple search aims for the highest log p
    @property
    def best_objective(self):
        return self.best_log

    # save how this round went
    def record_round(self, mean_loss):
        if not self.config.verbose and self.on_step is None:
            return
        point = dict(step=self.steps, mean_loss=mean_loss, best_log_probability=self.best_log, best_ned=self.best_ned, best_vienna_probability=self.best_vienna_probability, best_vienna_ned=self.best_vienna_ned)
        if self.config.verbose:
            self.history.append(point)
        if self.on_step:
            self.on_step(point)

    def run(self):
        started = time.perf_counter()
        losses, best_average, stale = [], math.inf, 0
        best_seen = -math.inf
        self.score_argmax()
        stop = 'round_limit'

        try:
            for step in range(1, self.config.max_rounds + 1):
                self.steps = step

                # learn from this batch, then try the most likely sequence
                assignments, sequences = self.sample()
                values = self.score(sequences)
                losses.append(-float(np.mean(values)))
                self.update(assignments, values)
                self.score_argmax()

                # see whether our best sequence or recent average got better
                if self.best_objective > best_seen:
                    self.best_stale_steps, self.best_found_step = 0, step
                else:
                    self.best_stale_steps += 1
                best_seen = max(best_seen, self.best_objective)
                if len(losses) >= self.config.patience:
                    average = float(np.mean(losses[-self.config.patience:]))
                    if average < best_average:
                        best_average, stale = average, 0
                    else:
                        stale += 1

                self.objective_stale_steps = stale
                self.record_round(losses[-1])

                # call it a day when both the recent average and best objective stop improving
                if stale >= self.config.patience and self.best_stale_steps >= self.config.patience:
                    stop = 'plateau'
                    break
        except RequestLimit:
            stop = 'request_limit'

        return self.build_result(stop, started)

    # compare the finalists with every argmax already checked by vienna
    def build_result(self, stop, started):
        finalists = self.check_vienna((self.best_log_sequence, self.best_ned_sequence, self.argmax()))
        validation_cpu = self.vienna_cpu_seconds
        best_p = max(finalists, key=lambda sequence: finalists[sequence][1])
        best_ned = min(finalists, key=lambda sequence: finalists[sequence][0].ned)
        score, probability = finalists[best_p]
        is_mfe, is_umfe, structure_cpu = validate_structure(best_p, self.target)
        validation_cpu += structure_cpu
        result = dict(
            sequence=best_p,
            probability=probability,
            log_probability=score.logp,
            best_ned=finalists[best_ned][0].ned,
            best_ned_sequence=best_ned,
            sequence_ned=score.ned,
            is_mfe=is_mfe,
            is_umfe=is_umfe,
            best_approximate_probability=math.exp(self.best_log),
            best_approximate_ned=self.best_ned,
            steps=self.steps,
            requests=self.requests,
            unique_search=len(self.scorer.cache),
            unique_vienna=len(finalists),
            stop=stop,
            score_cpu_seconds=self.scorer.cpu_seconds,
            validation_cpu_seconds=validation_cpu,
            runtime_seconds=time.perf_counter() - started,
            scoring_engine='lazyoutside',
            final_validation_engine='vienna',
            scores_are_approximate=False,
            search_scores_are_approximate=True,
            config=asdict(self.config),
        )
        if self.config.verbose:
            result['history'] = self.history
        return result


# design one target with lazyoutside scores and 1-ned weights
def optimize(target, *, config=None, workers=1, on_step=None, **settings):
    config = config or Config(**settings)
    with ScoringPool(workers) as pool:
        return CEM(target, config, pool=pool, on_step=on_step).run()


# let targets and seeds share the same scoring workers
def optimize_many(targets, *, config=None, workers=1, concurrent_puzzles=None, seeds=None, **settings):
    from .parallel.batch import optimize_targets
    return optimize_targets(targets, config=config or Config(**settings), workers=workers, concurrent_puzzles=concurrent_puzzles, seeds=seeds)
