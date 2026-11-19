import csv
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Sample:
    sample_id: str
    group_id: str
    split: str
    live: bool
    scores: tuple


@dataclass(frozen=True)
class ScoreTable:
    models: tuple
    samples: tuple

    def partition(self, name):
        rows = [sample for sample in self.samples if sample.split == name]
        if {sample.live for sample in rows} != {True, False}:
            raise ValueError(f'{name} needs both live and spoof samples')
        return rows


def live_probability(probabilities, live_index=1):
    values = tuple(float(value) for value in probabilities)
    if not 0 <= live_index < len(values) or len(values) < 2:
        raise ValueError('expected a class probability vector with a live class')
    if not all(isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError('invalid class probability')
    if abs(sum(values) - 1) > 1e-4:
        raise ValueError('class probabilities must sum to one')
    return values[live_index]


def read_table(stream):
    reader = csv.reader(stream, strict=True)
    header = next(reader, [])
    if header[:4] != ['sample_id', 'group_id', 'split', 'label'] or len(header) < 5:
        raise ValueError('expected sample_id,group_id,split,label and one or more model columns')
    if any(not name or name != name.strip() for name in header) or len(header) != len(set(header)):
        raise ValueError('column names must be nonempty, trimmed and unique')
    samples, sample_ids, group_splits = [], set(), {}
    for row in reader:
        if len(row) != len(header):
            raise ValueError(f'row {reader.line_num} has the wrong column count')
        sample_id, group_id, split, label = row[:4]
        if any(not value or value != value.strip() for value in (sample_id, group_id)):
            raise ValueError(f'row {reader.line_num} needs trimmed sample and group IDs')
        if sample_id in sample_ids or group_splits.setdefault(group_id, split) != split:
            raise ValueError(f'row {reader.line_num} repeats a sample or shares a group across splits')
        if split not in ('train', 'dev', 'test') or label not in ('live', 'spoof'):
            raise ValueError(f'row {reader.line_num} has an unknown split or label')
        try:
            scores = tuple(float(value) for value in row[4:])
        except ValueError:
            raise ValueError(f'row {reader.line_num} contains a nonnumeric score') from None
        if not all(isfinite(value) and 0 <= value <= 1 for value in scores):
            raise ValueError(f'row {reader.line_num} scores must be probabilities in [0, 1]')
        samples.append(Sample(sample_id, group_id, split, label == 'live', scores))
        sample_ids.add(sample_id)
    return ScoreTable(tuple(header[4:]), tuple(samples))
