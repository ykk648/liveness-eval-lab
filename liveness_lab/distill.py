"""Distill a compact classifier using logits or within-batch relations."""

import argparse
from math import isfinite

import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader, TensorDataset

from .compact import CompactNet, load_model


def kd_loss(student, teacher, labels, temperature=2.0, soft_weight=0.7):
    if not isfinite(temperature) or temperature <= 0 or not 0 <= soft_weight <= 1:
        raise ValueError('expected positive temperature and soft weight in [0, 1]')
    soft = functional.kl_div(functional.log_softmax(student / temperature, dim=1),
                             functional.softmax(teacher.detach() / temperature, dim=1), reduction='batchmean')
    return soft_weight * temperature**2 * soft + (1 - soft_weight) * functional.cross_entropy(student, labels)


def relations(outputs):
    offsets = outputs.unsqueeze(1) - outputs.unsqueeze(0)
    distances = torch.linalg.vector_norm(offsets, dim=-1)
    mean = distances.sum() / distances.gt(0).sum().clamp_min(1)
    directions = functional.normalize(offsets, dim=-1)
    angles = directions @ directions.transpose(1, 2)
    return distances / mean.clamp_min(1e-12), angles


def rkd_loss(student, teacher):
    student_distance, student_angle = relations(student)
    teacher_distance, teacher_angle = relations(teacher.detach())
    return (functional.smooth_l1_loss(student_distance, teacher_distance) +
            2 * functional.smooth_l1_loss(student_angle, teacher_angle))


def train_epoch(student, teacher, batches, optimizer, method):
    if method not in ('kd', 'rkd') or student is teacher:
        raise ValueError('expected distinct teacher and student and method kd or rkd')
    student.train()
    teacher.eval()
    accumulated = 0.0
    total = 0
    for images, labels in batches:
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            targets = teacher(images)
        predictions = student(images)
        loss = (kd_loss(predictions, targets, labels) if method == 'kd'
                else rkd_loss(predictions, targets))
        loss.backward()
        optimizer.step()
        total += images.shape[0]
        accumulated += loss.item() * images.shape[0]
    if total == 0:
        raise ValueError('no full training batches')
    return accumulated / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('kd', 'rkd'), required=True)
    parser.add_argument('--preset', choices=('compact-a', 'compact-b'), default='compact-a')
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', help='tensor dictionary with images [N,3,80,80] and labels [N]')
    parser.add_argument('--teacher', help='trained CompactNet checkpoint for a real dataset')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 3 or bool(args.data) != bool(args.teacher):
        parser.error('use positive epochs, batch-size >= 3 and supply data and teacher together')
    torch.manual_seed(17)
    torch.set_num_threads(1)
    if args.data:
        tensors = torch.load(args.data, map_location='cpu', weights_only=True)
        images, labels = tensors['images'], tensors['labels']
        teacher = load_model(args.teacher)
    else:
        images = torch.randn(args.batch_size * 2, 3, 80, 80)
        labels = torch.arange(len(images)) % 3
        teacher = CompactNet('compact-b')
    classes = teacher.logits.out_features
    if (images.shape[1:] != (3, 80, 80) or images.dtype != torch.float32 or
            labels.shape != (len(images),) or labels.dtype != torch.long or len(images) < args.batch_size or
            labels.min() < 0 or labels.max() >= classes):
        parser.error('expected float32 80x80 RGB crops, int64 class labels and at least one full batch')
    batches = DataLoader(TensorDataset(images, labels), batch_size=args.batch_size, shuffle=True, drop_last=True)
    student = CompactNet(args.preset, classes)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(args.epochs):
        loss = train_epoch(student, teacher, batches, optimizer, args.method)
        print(f'epoch={epoch + 1} method={args.method} loss={loss:.6f}')
    torch.save({'preset': args.preset, 'classes': classes, 'state_dict': student.state_dict()}, args.output)


if __name__ == '__main__':
    main()
