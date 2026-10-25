# keep track of progress, structure checks, and each score's winner
import math


STRUCTURAL_METRICS = (
    'energy_gap_kcal_mol',
    'mfe_structure_distance',
    'is_mfe',
    'is_umfe',
)
METRICS_TO_MAXIMIZE = {'probability', 'is_mfe', 'is_umfe'}


# keep the notes separate from the search itself
class SearchTracking:
    def __init__(self, config):
        self.config = config
        self.history = []
        self.metric_winners = {}

        # remember the first mfe and umfe solutions we find
        self.mfe_sequence = None
        self.umfe_sequence = None
        self.umfe_found_step = None
        self.structure_checked = set()
        self.structure_cpu_seconds = 0.0

    # count each structure check once and keep any new winners
    def observe(self, sequence, value, step):
        has_structure_check = value.is_umfe is not None
        if has_structure_check and sequence not in self.structure_checked:
            self.structure_checked.add(sequence)
            self.structure_cpu_seconds += value.mfe_cpu_seconds

            if value.is_mfe and self.mfe_sequence is None:
                self.mfe_sequence = sequence
            if value.is_umfe and self.umfe_sequence is None:
                self.umfe_sequence = sequence
                self.umfe_found_step = step

        if value.metrics is not None:
            for name in STRUCTURAL_METRICS:
                self.record_metric(name, value.metrics[name], sequence, 'vienna', step, value.metrics)

    # only replace a winner when the new score is better
    def record_metric(self, name, value, sequence, source, step, metrics=None):
        previous = self.metric_winners.get(name)
        if previous is None:
            improved = True
        elif name in METRICS_TO_MAXIMIZE:
            improved = value > previous['value']
        else:
            improved = value < previous['value']

        if improved:
            self.metric_winners[name] = {
                'value': value,
                'sequence': sequence,
                'source': source,
                'metrics': metrics,
            }
            if self.config.verbose:
                self.metric_winners[name]['step'] = step

    # remember this sample's p and ned from the search scorer
    def observe_search(self, sequence, value, step, backend):
        self.observe(sequence, value, step)
        if not self.config.track_all:
            return

        probability = value.probability
        if probability is None:
            probability = math.exp(value.logp)
        self.record_metric('probability', probability, sequence, backend, step)
        if value.ned is not None:
            self.record_metric('normalized_ensemble_defect', value.ned, sequence, backend, step)

    # check the selected elites when extra tracking is on
    def observe_elites(self, sequences, scorer, step, *, track_ned):
        if not self.config.track_all:
            return

        scores = scorer.score_structures(sequences)
        for sequence, value in zip(sequences, scores):
            self.observe(sequence, value, step)
            if track_ned:
                self.record_metric('normalized_ensemble_defect', value.ned, sequence, 'vienna_elites', step)

    # save this round's progress and share it with the callback
    def record_round(self, search, mean_loss):
        if not self.config.verbose and search.on_step is None:
            return
        structure_tracking_enabled = self.config.track_umfe or self.config.track_all
        best_ned = search.best_ned if math.isfinite(search.best_ned) else None
        point = {
            'step': search.steps,
            'mean_loss': mean_loss,
            'best_objective': search.best_value,
            'best_log_probability': search.best_log,
            'best_search_ned': best_ned,
            'umfe_found': bool(self.umfe_sequence) if structure_tracking_enabled else None,
            'umfe_found_step': self.umfe_found_step,
            'structure_checked_unique': len(self.structure_checked),
        }
        if self.config.track_all:
            point['best_metrics'] = {name: winner['value'] for name, winner in self.metric_winners.items()}

        if self.config.verbose:
            self.history.append(point)
        if search.on_step:
            search.on_step(point)
