import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from cli import PROJECT_ROOT, add_retrieval_arguments
from data_utils import CIRRDataset
from qwen_encoder import QwenEmbeddingEncoder
from retrieval import load_or_extract_index_features, rank_queries


def generate_predictions(
    dataset: CIRRDataset,
    encoder: QwenEmbeddingEncoder,
    index_features: torch.Tensor,
    index_names: list[str],
    batch_size: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    predictions = {}
    group_predictions = {}
    rankings = rank_queries(dataset, encoder, index_features, index_names, batch_size)
    for item, ranked_names, group_names in rankings:
        pair_id = str(item["pair_id"])
        predictions[pair_id] = ranked_names
        group_predictions[pair_id] = group_names

    return predictions, group_predictions


def generate_submission(args) -> tuple[Path, Path]:
    encoder = QwenEmbeddingEncoder(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        query_instruction=args.query_instruction,
        adapter_path=args.adapter,
    )
    classic_dataset = CIRRDataset("test1", "classic", args.cirr_root)
    relative_dataset = CIRRDataset("test1", "relative", args.cirr_root)
    index_features, index_names = load_or_extract_index_features(
        classic_dataset,
        encoder,
        args.batch_size,
        args.index_cache,
    )

    predictions, group_predictions = generate_predictions(
        relative_dataset,
        encoder,
        index_features,
        index_names,
        args.batch_size,
    )
    submission = {"version": "rc2", "metric": "recall", **predictions}
    group_submission = {
        "version": "rc2",
        "metric": "recall_subset",
        **group_predictions,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_path = args.output_dir / f"recall_submission_{args.name}.json"
    group_path = args.output_dir / f"recall_subset_submission_{args.name}.json"
    with global_path.open("w") as file:
        json.dump(submission, file, sort_keys=True)
    with group_path.open("w") as file:
        json.dump(group_submission, file, sort_keys=True)
    return global_path, group_path


def parse_args():
    parser = ArgumentParser(
        description="Create a CIRR test submission with Qwen3-VL-Embedding"
    )
    add_retrieval_arguments(parser, "test1")
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "submission/CIRR",
    )
    return parser.parse_args()


if __name__ == "__main__":
    paths = generate_submission(parse_args())
    print("\n".join(map(str, paths)))
