import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'requires the train extra')
class CompactTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_pruned_profiles(self):
        from liveness_lab.compact import CompactNet, channel_config

        inputs = torch.randn(2, 3, 80, 80)
        for preset, count in [('compact-a', 273058), ('compact-b', 281497)]:
            model = CompactNet(preset).eval()
            self.assertEqual(len(channel_config(preset)), 49)
            self.assertEqual(sum(param.numel() for param in model.parameters()), count)
            self.assertEqual(tuple(model(inputs).shape), (2, 3))

    def test_student_updates_without_changing_teacher(self):
        from liveness_lab.compact import CompactNet
        from liveness_lab.distill import train_epoch

        for method in ('kd', 'rkd'):
            student, teacher = CompactNet(), CompactNet('compact-b')
            frozen = {key: value.clone() for key, value in teacher.state_dict().items()}
            before = student.logits.weight.detach().clone()
            optimizer = torch.optim.SGD(student.parameters(), lr=.01)
            loss = train_epoch(student, teacher, [(torch.randn(4, 3, 80, 80), torch.tensor([0, 1, 2, 1]))], optimizer, method)
            self.assertGreater(loss, 0)
            self.assertFalse(torch.equal(student.logits.weight, before))
            self.assertTrue(all(torch.equal(value, frozen[key]) for key, value in teacher.state_dict().items()))

    def test_degenerate_relations_are_finite(self):
        from liveness_lab.distill import rkd_loss

        outputs = torch.zeros(3, 3, requires_grad=True)
        loss = rkd_loss(outputs, torch.ones(3, 3))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(outputs.grad).all())

    @unittest.skipUnless(os.environ.get('CAFFE_PROTO') and os.environ.get('CAFFE2NCNN'), 'requires conversion tools')
    def test_three_backend_parity(self):
        import subprocess
        from liveness_lab.compact import CompactNet
        from liveness_lab.export import verify, write_caffe

        for preset in ('compact-a', 'compact-b'):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                model = CompactNet(preset).eval()
                write_caffe(model, os.environ['CAFFE_PROTO'], work)
                subprocess.run([os.environ['CAFFE2NCNN'], 'model.prototxt', 'model.caffemodel',
                                'model.param', 'model.bin'], cwd=work, check=True)
                self.assertEqual(set(verify(model, work, True)), {'caffe', 'ncnn'})
