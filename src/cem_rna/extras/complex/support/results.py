# check the finalists with vienna and collect the results
from dataclasses import asdict
import math
import time
from ....validation import validate_structure


# bring together the final scores, search details, and any structure checks
def build_result(search, stop, started):
    exact_scores = _validate_candidates(search)

    # keep the same tie rules: lazyoutside compares p directly,
    # the other scoring choices compare log p
    use_raw_probability = search.mode.backend == 'lazy'
    if use_raw_probability:
        best_probability_sequence = max(exact_scores, key=lambda sequence: exact_scores[sequence].probability)
    else:
        best_probability_sequence = max(exact_scores, key=lambda sequence: exact_scores[sequence].logp)
    best_ned_sequence = min(exact_scores, key=lambda sequence: exact_scores[sequence].ned)

    probability_score = exact_scores[best_probability_sequence]
    ned_score = exact_scores[best_ned_sequence]
    probability = probability_score.probability if use_raw_probability else math.exp(probability_score.logp)
    is_mfe, is_umfe, structure_cpu = validate_structure(best_probability_sequence, search.target, probability_score)
    validation_cpu_seconds = (0.0 if search.vienna is search.scorer else search.vienna.cpu_seconds) + structure_cpu

    result = {
        'sequence': best_probability_sequence,
        'probability': probability,
        'log_probability': probability_score.logp,
        'best_ned': ned_score.ned,
        'best_ned_sequence': best_ned_sequence,
        'sequence_ned': probability_score.ned,
        'is_mfe': is_mfe,
        'is_umfe': is_umfe,
        'steps': search.steps,
        'requests': search.requests,
        'stop': stop,
        'unique_search': len(search.scorer.cache),
        'unique_vienna': len(exact_scores),
        'search_cpu_seconds': search.scorer.cpu_seconds,
        'validation_cpu_seconds': validation_cpu_seconds,
        'runtime_seconds': time.perf_counter() - started,
        'config': asdict(search.config),
        'sample_structures': _structure_summary(search),
    }

    if search.config.verbose:
        result.update(history=search.tracking.history, best_found_step=search.best_found_step,
                      best_stale_steps=search.best_stale_steps, objective_stale_steps=search.objective_stale_steps)
        if search.config.track_all:
            result['tracking_scope'] = _tracking_scope(search.mode.backend)

    if search.config.track_all:
        result['metric_winners'] = search.tracking.metric_winners

    if search.config.track_sep or search.config.track_all:
        metric_started = time.process_time()
        result['metrics'] = _winner_metrics(search, best_probability_sequence, best_ned_sequence)
        result['final_metrics_cpu_seconds'] = time.process_time() - metric_started
        result['runtime_seconds'] = time.perf_counter() - started

    return result


# score the finalists and keep any elites we checked earlier
def _validate_candidates(search):
    shortlist = (
        search.best_sequence,
        search.best_ned_sequence,
        search.best_log_sequence,
        search.argmax(),
    )
    # skip repeats and keep the order so the first sequence wins a tie
    candidates = list(dict.fromkeys(sequence for sequence in shortlist if sequence))
    search.vienna.score(candidates)

    if search.config.track_all:
        scores = search.vienna.score_structures(candidates, observations=search.scorer.cache)
        for sequence, score in zip(candidates, scores):
            search.tracking.observe(sequence, score, search.steps)
            if search.best_ned_sequence is None:
                search.tracking.record_metric('normalized_ensemble_defect', score.ned, sequence, 'vienna_final', search.steps)

    # the hybrid search may already have vienna scores for earlier elites
    return search.vienna.cache


# say what we checked and which sequences first solved the target
def _structure_summary(search):
    tracking = search.tracking
    enabled = search.config.track_umfe or search.config.track_all
    if search.config.track_all and not search.config.track_umfe:
        scope = 'Vienna elites and final candidates'
    else:
        scope = 'all unique search candidates until the first exact uMFE witness'

    summary = {
        'enabled': enabled,
        'scope': scope,
        'is_mfe': bool(tracking.mfe_sequence) if enabled else None,
        'is_umfe': bool(tracking.umfe_sequence) if enabled else None,
        'mfe_sequence': tracking.mfe_sequence,
        'umfe_sequence': tracking.umfe_sequence,
        'checked_unique': len(tracking.structure_checked),
        'cpu_seconds': tracking.structure_cpu_seconds,
    }
    if search.config.verbose:
        summary['umfe_found_step'] = tracking.umfe_found_step
    return summary


# say which candidates each score came from
def _tracking_scope(backend):
    if backend in ('lazy', 'vienna'):
        ned_scope = 'All unique search candidates, using the selected scoring backend'
    else:
        ned_scope = 'Vienna elites/final candidates only'

    return {
        'probability': 'All unique search candidates, using the selected scoring backend',
        'normalized_ensemble_defect': ned_scope,
        'structural_metrics': ('Exact Vienna MFE, energy gap, distance and uniqueness ' 'on elites and final candidates only'),
    }


# if both winners are the same sequence, check its structure just once
def _winner_metrics(search, probability_sequence, ned_sequence):
    winners = list(dict.fromkeys((probability_sequence, ned_sequence)))
    scores = search.vienna.score_structures(winners, observations=search.scorer.cache)
    metrics_by_sequence = {}
    for sequence, score in zip(winners, scores):
        metrics_by_sequence[sequence] = score.metrics
        if search.config.track_all:
            search.tracking.observe(sequence, score, search.steps)

    return {'best_p': metrics_by_sequence[probability_sequence], 'best_ned': metrics_by_sequence[ned_sequence]}
