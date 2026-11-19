from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, groupby
from math import comb, fsum
from typing import Optional


@dataclass(frozen=True)
class Point:
    threshold: Optional[float]
    accepted_live: int
    accepted_spoof: int
    live_count: int
    spoof_count: int

    def report(self):
        return dict(threshold=self.threshold, accepted_live=self.accepted_live,
                    live_count=self.live_count, accepted_spoof=self.accepted_spoof,
                    spoof_count=self.spoof_count, tpr=self.accepted_live / self.live_count,
                    fpr=self.accepted_spoof / self.spoof_count)


def fused(sample, columns):
    return fsum(sample.scores[column] for column in columns) / len(columns)


def operating_points(samples, columns):
    live_count = sum(sample.live for sample in samples)
    spoof_count = len(samples) - live_count
    yield Point(None, 0, 0, live_count, spoof_count)
    accepted_live = accepted_spoof = 0
    ordered = sorted(((fused(sample, columns), sample.live) for sample in samples), reverse=True)
    for threshold, tied in groupby(ordered, key=lambda pair: pair[0]):
        for _, live in tied:
            if live:
                accepted_live += 1
            else:
                accepted_spoof += 1
        yield Point(threshold, accepted_live, accepted_spoof, live_count, spoof_count)


def strict_key(point):
    return point.accepted_live, -point.accepted_spoof


def calibrate(samples, columns, target):
    points = operating_points(samples, columns)
    strict = nearest = next(points)
    budget = target * strict.spoof_count
    for point in points:
        if point.accepted_spoof <= budget and strict_key(point) > strict_key(strict):
            strict = point
        candidate_distance = abs(point.accepted_spoof - budget)
        nearest_distance = abs(nearest.accepted_spoof - budget)
        if (candidate_distance, point.accepted_spoof, -point.accepted_live) < (
                nearest_distance, nearest.accepted_spoof, -nearest.accepted_live):
            nearest = point
    return strict, nearest


def accepts(score, threshold):
    return threshold is not None and score >= threshold


def apply_threshold(samples, columns, threshold):
    accepted = [sample for sample in samples if accepts(fused(sample, columns), threshold)]
    accepted_live = sum(sample.live for sample in accepted)
    live_count = sum(sample.live for sample in samples)
    return Point(threshold, accepted_live, len(accepted) - accepted_live, live_count, len(samples) - live_count)


def evaluate(table, target_fpr, max_size):
    target = Fraction(str(target_fpr))
    if not 0 <= target <= 1 or not 1 <= max_size <= len(table.models):
        raise ValueError('check target FPR [0, 1] and ensemble size [1, model count]')
    candidate_count = sum(comb(len(table.models), size) for size in range(1, max_size + 1))
    if candidate_count > 10000:
        raise ValueError('over 10000 combinations; reduce max-size or shortlist the models')
    dev = table.partition('dev')
    best = baseline = None
    order = sorted(range(len(table.models)), key=table.models.__getitem__)
    for size in range(1, max_size + 1):
        for columns in combinations(order, size):
            strict, nearest = calibrate(dev, columns, target)
            if best is None or strict_key(strict) > strict_key(best[1]):
                best = columns, strict, nearest
        if size == 1:
            baseline = best
    held_out = table.partition('test')

    def describe(candidate):
        columns, strict, nearest = candidate
        return {'models': [table.models[index] for index in columns], 'dev': strict.report(),
                'dev_nearest_fpr': nearest.report(),
                'test': apply_threshold(held_out, columns, strict.threshold).report()}

    return {'target_fpr': str(target), 'candidate_count': candidate_count,
            'best_single': describe(baseline), 'selected': describe(best)}


def mine(table, model, per_class):
    if per_class < 1:
        raise ValueError('per-class must be positive')
    column = table.models.index(model)
    rows = table.partition('train')
    result = {}
    for label, live in [('live', True), ('spoof', False)]:
        subset = (sample for sample in rows if sample.live == live)
        ranked = sorted(subset, key=lambda sample: (
            sample.scores[column] if live else -sample.scores[column], sample.sample_id))
        result[label] = [sample.sample_id for sample in ranked[:per_class]]
    return result


def parity(table, reference, converted, threshold):
    if not 0 <= threshold <= 1 or reference == converted:
        raise ValueError('use two distinct backend columns and a threshold in [0, 1]')
    ref_col, new_col = table.models.index(reference), table.models.index(converted)
    rows = table.partition('test')
    deltas, flips = [], {'live_rejected': 0, 'live_accepted': 0, 'spoof_rejected': 0, 'spoof_accepted': 0}
    for sample in rows:
        before, after = sample.scores[ref_col], sample.scores[new_col]
        deltas.append(abs(after - before))
        if accepts(before, threshold) != accepts(after, threshold):
            label = 'live' if sample.live else 'spoof'
            decision = 'accepted' if after >= threshold else 'rejected'
            flips[f'{label}_{decision}'] += 1
    return {'rows': len(rows), 'threshold': threshold, 'mean_abs_delta': fsum(deltas) / len(rows),
            'max_abs_delta': max(deltas), 'decision_flips': flips}
