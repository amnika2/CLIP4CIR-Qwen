import copy
import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from contrastive import contrastive_loss
from train_lora import collate_items, train_epoch


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_vectors = torch.nn.Embedding.from_pretrained(
            torch.tensor([[1.0, 0.3], [0.2, 1.0]]), freeze=False
        )
        self.target_vectors = torch.nn.Embedding.from_pretrained(
            torch.tensor([[0.7, 0.1], [0.3, 0.8]]), freeze=False
        )
        self.model = SimpleNamespace(device=torch.device("cpu"))

    def forward_queries(self, images, captions):
        return F.normalize(self.query_vectors(torch.tensor(images)), dim=-1)

    def forward_images(self, images):
        return F.normalize(self.target_vectors(torch.tensor(images)), dim=-1)


class TrainingTests(unittest.TestCase):
    def test_contrastive_loss_prefers_matched_pairs_and_backpropagates(self):
        queries = torch.eye(3, requires_grad=True)
        targets = torch.eye(3, requires_grad=True)
        matched = contrastive_loss(queries, targets, logit_scale=5)
        mismatched = contrastive_loss(queries, targets.roll(1, 0), logit_scale=5)
        self.assertLess(matched.item(), mismatched.item())
        matched.backward()
        for features in (queries, targets):
            self.assertTrue(torch.isfinite(features.grad).all())
            self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_partial_accumulation_group_matches_manual_updates(self):
        for num_batches in (1, 2, 3, 4, 5):
            with self.subTest(num_batches=num_batches):
                encoder = TinyEncoder()
                expected_encoder = copy.deepcopy(encoder)
                items = [
                    {"reference_image": i, "caption": "", "target_image": i}
                    for _ in range(num_batches)
                    for i in (0, 1)
                ]
                loader = DataLoader(items, batch_size=2, collate_fn=collate_items)
                args = SimpleNamespace(
                    epochs=1,
                    dtype="float32",
                    logit_scale=2.0,
                    gradient_accumulation_steps=2,
                    max_grad_norm=math.inf,
                )
                optimizer = SGD(encoder.parameters(), lr=0.1)
                scheduler = LambdaLR(optimizer, lambda step: 1)
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                loss = train_epoch(
                    encoder, loader, optimizer, scheduler, scaler, args, 1
                )

                expected_optimizer = SGD(expected_encoder.parameters(), lr=0.1)
                for _ in range(math.ceil(num_batches / 2)):
                    expected_optimizer.zero_grad()
                    expected_loss = contrastive_loss(
                        expected_encoder.forward_queries([0, 1], ["", ""]),
                        expected_encoder.forward_images([0, 1]),
                        logit_scale=2,
                    )
                    expected_loss.backward()
                    expected_optimizer.step()

                self.assertTrue(math.isfinite(loss))
                self.assertEqual(scheduler.last_epoch, math.ceil(num_batches / 2))
                for actual, expected in zip(
                    encoder.parameters(), expected_encoder.parameters()
                ):
                    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
