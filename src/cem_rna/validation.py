# let vienna check the final p and ned
import time
from .scorer import Score

CALORIES_PER_KILOCALORIE = 1000
CENTIKCAL_PER_KCAL = 100


# calculate exact p and optional ned, then let extras reuse the fold for structure checks
def fold_sequence(sequence, target, *, compute_ned=True):
    import RNA
    model = RNA.md()
    model.compute_bpp = int(compute_ned)
    fold = RNA.fold_compound(sequence, model)
    structure, mfe = fold.mfe()
    fold.exp_params_rescale(mfe)
    _, ensemble = fold.pf()
    energy = fold.eval_structure(target)
    rt = RNA.GASCONST * (model.temperature + RNA.K0) / CALORIES_PER_KILOCALORIE
    logp = min(0.0, -(energy - ensemble) / rt)
    ned = float(fold.ensemble_defect(target)) if compute_ned else None
    if ned is not None:
        ned = min(1.0, max(0.0, ned))
    return Score(float(logp), ned), float(fold.pr_structure(target)), fold, structure, mfe


# only validate each final sequence once, even when p and ned pick the same winner
def validate_finalists(target, sequences):
    started = time.process_time()
    scores = {}
    for sequence in dict.fromkeys(sequences):
        score, probability, _, _, _ = fold_sequence(sequence, target)
        scores[sequence] = score, probability
    return scores, time.process_time() - started


# check mfe and umfe without calculating the partition function or ned
def mfe_status(sequence, target, fold=None, structure=None, mfe=None):
    import RNA

    if fold is None:
        fold = RNA.fold_compound(sequence)
    if structure is None:
        structure, mfe = fold.mfe()
    # compare whole energy units so tiny rounding errors do not fool us
    target_energy_units = round(float(fold.eval_structure(target)) * CENTIKCAL_PER_KCAL)
    minimum_energy_units = round(float(mfe) * CENTIKCAL_PER_KCAL)
    is_mfe = target_energy_units == minimum_energy_units
    is_umfe = False
    if is_mfe and structure == target:
        # check for equal energy rivals even when the mfe structure matches
        model_settings = RNA.md()
        model_settings.uniq_ML = 1
        unique_fold = RNA.fold_compound(sequence, model_settings)
        unique_fold.mfe()
        seen = {'target': False, 'other': False}

        # vienna sends none when it has finished listing structures
        def visit(dot_bracket, energy, data):
            if dot_bracket is not None:
                seen['target' if dot_bracket == target else 'other'] = True

        unique_fold.subopt_cb(0, visit)
        is_umfe = seen['target'] and not seen['other']
    return is_mfe, is_umfe


# reuse a saved exact check or check this final sequence once
def validate_structure(sequence, target, score=None):
    if score is not None and score.is_umfe is not None:
        return score.is_mfe, score.is_umfe, 0.0
    started = time.process_time()
    is_mfe, is_umfe = mfe_status(sequence, target)
    return is_mfe, is_umfe, time.process_time() - started
