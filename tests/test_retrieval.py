import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cirr_test_submission import generate_predictions, generate_submission
from data_utils import CIRRDataset
from retrieval import compute_cirr_metrics, load_or_extract_index_features, rank_queries


class FeatureEncoder:
    model_name = "test-model"
    dtype = "float32"

    def __init__(self, query_features):
        self.query_features = query_features
        self.image_calls = 0

    def encode_queries(self, images, captions, batch_size):
        return self.query_features[[int(caption) for caption in captions]]

    def encode_images(self, images, batch_size, show_progress):
        self.image_calls += 1
        return torch.eye(len(images))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        annotations = self.root / "cirr"
        (annotations / "captions").mkdir(parents=True)
        (annotations / "image_splits").mkdir()
        self.names = [f"image_{i}" for i in range(60)]
        images = {name: f"images/{name}.png" for name in self.names}
        groups = [[0, 1, 5, 10, 20, 30], [1, 2, 5, 10, 20, 30], [2, 55, 56, 57, 58, 59]]
        targets = [1, 2, 55]
        self.triplets = [
            {
                "reference": self.names[i],
                "target_hard": self.names[targets[i]],
                "caption": str(i),
                "pairid": i + 100,
                "img_set": {"members": [self.names[j] for j in group]},
            }
            for i, group in enumerate(groups)
        ]
        for split in ("train", "val", "test1"):
            triplets = [dict(triplet) for triplet in self.triplets]
            if split == "test1":
                for triplet in triplets:
                    del triplet["target_hard"]
            (annotations / "captions" / f"cap.rc2.{split}.json").write_text(
                json.dumps(triplets)
            )
            (annotations / "image_splits" / f"split.rc2.{split}.json").write_text(
                json.dumps(images)
            )

        self.features = torch.eye(60)
        self.queries = torch.arange(60, 0, -1, dtype=torch.float32).repeat(3, 1)
        self.queries[torch.arange(3), torch.arange(3)] = 100
        self.encoder = FeatureEncoder(self.queries)

    def test_dataset_modes_and_splits(self):
        classic = CIRRDataset("val", "classic", self.root)
        self.assertEqual(len(classic), 60)
        self.assertEqual(classic[0]["image_name"], "image_0")
        for split in ("train", "val"):
            relative = CIRRDataset(split, "relative", self.root)
            self.assertEqual(len(relative), 3)
            self.assertEqual(relative[0]["target_name"], "image_1")
            self.assertEqual(
                relative[0]["target_image"], str(self.root / "images/image_1.png")
            )
        test_item = CIRRDataset("test1", "relative", self.root)[0]
        self.assertEqual(test_item["pair_id"], 100)
        self.assertNotIn("target_name", test_item)

    def test_topk_matches_full_ranking_and_excludes_reference(self):
        dataset = CIRRDataset("val", "relative", self.root)
        for batch_size in (1, 2, 3):
            with self.subTest(batch_size=batch_size):
                rankings = list(
                    rank_queries(
                        dataset, self.encoder, self.features, self.names, batch_size
                    )
                )
                self.assertEqual(len(rankings), 3)
                for i, (item, ranked, group_ranked) in enumerate(rankings):
                    expected = [
                        self.names[j]
                        for j in self.queries[i].argsort(descending=True).tolist()
                    ]
                    expected.remove(item["reference_name"])
                    self.assertEqual(ranked, expected[:50])
                    self.assertEqual(
                        group_ranked,
                        [name for name in expected if name in item["group_members"]][
                            :3
                        ],
                    )
                    self.assertNotIn(item["reference_name"], ranked + group_ranked)

    def test_group_recall_uses_entire_group(self):
        dataset = CIRRDataset("val", "relative", self.root)
        metrics = compute_cirr_metrics(
            rank_queries(dataset, self.encoder, self.features, self.names, 2)
        )
        self.assertAlmostEqual(metrics["recall_at1"], 100 / 3)
        for k in (5, 10, 50):
            self.assertAlmostEqual(metrics[f"recall_at{k}"], 200 / 3)
        for k in (1, 2, 3):
            self.assertEqual(metrics[f"group_recall_at{k}"], 100)

    def test_submission_uses_pair_ids_and_shared_ranking(self):
        dataset = CIRRDataset("test1", "relative", self.root)
        predictions, groups = generate_predictions(
            dataset, self.encoder, self.features, self.names, 2
        )
        self.assertEqual(set(predictions), {"100", "101", "102"})
        for pair_id in predictions:
            self.assertEqual(len(predictions[pair_id]), 50)
            self.assertEqual(len(groups[pair_id]), 3)
        self.assertEqual(groups["102"][0], "image_55")
        self.assertNotIn("image_55", predictions["102"])

    def test_cache_reuse_and_model_changes(self):
        dataset = CIRRDataset("val", "classic", self.root)
        cache_path = self.root / "cache/index.pt"
        first, names = load_or_extract_index_features(
            dataset, self.encoder, 2, cache_path
        )
        second, cached_names = load_or_extract_index_features(
            dataset, self.encoder, 2, cache_path
        )
        self.assertEqual(self.encoder.image_calls, 1)
        torch.testing.assert_close(first, second)
        self.assertEqual(names, cached_names)
        self.encoder.model_name = "test-model-with-adapter"
        load_or_extract_index_features(dataset, self.encoder, 2, cache_path)
        self.assertEqual(self.encoder.image_calls, 2)

    def test_submission_files_use_cirr_format(self):
        args = SimpleNamespace(
            model="test-model",
            device="cpu",
            dtype="float32",
            query_instruction="test",
            adapter=None,
            cirr_root=self.root,
            batch_size=2,
            index_cache=None,
            output_dir=self.root / "submission",
            name="test",
        )
        with patch(
            "cirr_test_submission.QwenEmbeddingEncoder", return_value=self.encoder
        ):
            global_path, group_path = generate_submission(args)
        self.assertEqual(global_path.name, "recall_submission_test.json")
        self.assertEqual(group_path.name, "recall_subset_submission_test.json")
        for path, metric, count in (
            (global_path, "recall", 50),
            (group_path, "recall_subset", 3),
        ):
            result = json.loads(path.read_text())
            self.assertEqual(result.pop("version"), "rc2")
            self.assertEqual(result.pop("metric"), metric)
            self.assertEqual(set(result), {"100", "101", "102"})
            self.assertTrue(all(len(names) == count for names in result.values()))


if __name__ == "__main__":
    unittest.main()
