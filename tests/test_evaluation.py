import io
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from liveness_lab.data import Sample, ScoreTable, live_probability, read_table
from liveness_lab.evaluation import calibrate, evaluate, mine, parity


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


def load_example(name):
    with (EXAMPLES / name).open(newline='') as stream:
        return read_table(stream)


class EvaluationTests(unittest.TestCase):
    def test_selection_and_frozen_holdout(self):
        table = load_example('ensemble.csv')
        report = evaluate(table, '.1', 2)
        self.assertEqual(report['candidate_count'], 3)
        self.assertEqual(report['selected']['models'], ['context', 'detail'])
        self.assertEqual(report['selected']['dev']['accepted_spoof'], 0)
        self.assertEqual(report['selected']['test']['accepted_spoof'], 1)
        self.assertEqual(report['selected']['test']['accepted_live'], 3)
        scrambled = tuple(replace(row, live=not row.live) if row.split == 'test' else row for row in table.samples)
        changed = evaluate(ScoreTable(table.models, scrambled), '.1', 2)['selected']
        self.assertEqual(changed['dev'], report['selected']['dev'])
        self.assertEqual(changed['models'], report['selected']['models'])

    def test_tied_scores_and_unattainable_target(self):
        rows = [Sample(str(index), str(index), 'dev', live, (score,))
                for index, (score, live) in enumerate([(0.9, True), (0.9, False), (0.3, False)])]
        strict, nearest = calibrate(rows, (0,), Fraction('0.4'))
        self.assertIsNone(strict.threshold)
        self.assertEqual((nearest.accepted_spoof, nearest.accepted_live), (1, 1))

    def test_tie_breaking_ignores_input_order(self):
        table = load_example('ensemble.csv')
        rows = tuple(replace(row, scores=(row.scores[0], row.scores[0])) for row in reversed(table.samples))
        result = evaluate(ScoreTable(('second', 'first'), rows), '0', 2)
        self.assertEqual(result['selected']['models'], ['first'])

    def test_rejects_leakage_and_invalid_rows(self):
        header = 'sample_id,group_id,split,label,model\n'
        first = 'd01,g01,dev,live,0.5\n'
        for bad in ('t01,g01,test,spoof,0.5', 'd01,g02,dev,live,0.1', 'd02,g02,dev,live,nan',
                    'd02,g02,dev,unknown,0.2', 'd02,g02,test,live,1.1', 'd02,g02,test,live', 'd02,,test,live,0.2'):
            with self.subTest(row=bad), self.assertRaises(ValueError):
                read_table(io.StringIO(header + first + bad))

    def test_missing_class_is_an_error(self):
        with self.assertRaises(ValueError):
            read_table(io.StringIO('sample_id,group_id,split,label,m\ns,g,test,live,0\n')).partition('test')

    def test_mining_uses_only_train_and_correct_score_direction(self):
        self.assertEqual(mine(load_example('ensemble.csv'), 'context', 1), {'live': ['tr02'], 'spoof': ['tr03']})

    def test_parity_counts_directional_flips(self):
        report = parity(load_example('parity.csv'), 'reference', 'converted', .7)
        self.assertEqual(report['decision_flips'], {'live_rejected': 1, 'live_accepted': 0,
                                                    'spoof_rejected': 0, 'spoof_accepted': 1})
        self.assertAlmostEqual(report['max_abs_delta'], .01)

    def test_upstream_probability_vector(self):
        self.assertEqual(live_probability([.7, .2, .1]), .2)
        for invalid in [[1.0], [0.1, 0.1], [-1, 2], [float('nan'), .5]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                live_probability(invalid)
