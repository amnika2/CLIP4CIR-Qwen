from pathlib import Path

import torch
from tqdm import tqdm

from data_utils import CIRRDataset
from qwen_encoder import QwenEmbeddingEncoder


def extract_index_features(
    dataset: CIRRDataset,
    encoder: QwenEmbeddingEncoder,
    batch_size: int,
) -> tuple[torch.Tensor, list[str]]:
    names = dataset.image_names
    paths = [dataset.image_path(name) for name in names]
    features = encoder.encode_images(paths, batch_size=batch_size, show_progress=True)
    return features, names


def load_or_extract_index_features(
    dataset: CIRRDataset,
    encoder: QwenEmbeddingEncoder,
    batch_size: int,
    cache_path: str | Path | None,
) -> tuple[torch.Tensor, list[str]]:
    if cache_path is None:
        return extract_index_features(dataset, encoder, batch_size)

    cache_path = Path(cache_path)
    metadata = {
        "model": encoder.model_name,
        "dtype": encoder.dtype,
        "root": str(dataset.root.resolve()),
        "split": dataset.split,
        "names": dataset.image_names,
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached["metadata"] == metadata:
            return cached["features"], cached["metadata"]["names"]

    features, names = extract_index_features(dataset, encoder, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "metadata": metadata}, cache_path)
    return features, names


def rank_queries(
    relative_dataset: CIRRDataset,
    encoder: QwenEmbeddingEncoder,
    index_features: torch.Tensor,
    index_names: list[str],
    batch_size: int,
):
    """Yield each query with its top 50 global and top 3 group candidates."""
    name_to_index = {name: index for index, name in enumerate(index_names)}
    batches = range(0, len(relative_dataset), batch_size)
    for start in tqdm(batches, desc=f"CIRR {relative_dataset.split} queries"):
        items = [
            relative_dataset[index]
            for index in range(start, min(start + batch_size, len(relative_dataset)))
        ]
        query_features = encoder.encode_queries(
            [item["reference_image"] for item in items],
            [item["caption"] for item in items],
            batch_size=batch_size,
        )
        scores = query_features @ index_features.T
        reference_indices = [name_to_index[item["reference_name"]] for item in items]
        scores[torch.arange(len(items)), reference_indices] = -torch.inf
        global_indices = scores.topk(50, dim=1).indices

        group_indices = torch.tensor(
            [[name_to_index[name] for name in item["group_members"]] for item in items]
        )
        group_scores = scores.gather(1, group_indices)
        group_order = group_scores.topk(3, dim=1).indices
        group_indices = group_indices.gather(1, group_order)

        for item, global_row, group_row in zip(items, global_indices, group_indices):
            ranked_names = [index_names[index] for index in global_row.tolist()]
            group_names = [index_names[index] for index in group_row.tolist()]
            yield item, ranked_names, group_names


def compute_cirr_metrics(rankings) -> dict[str, float]:
    metric_hits = {
        "group_recall_at1": 0,
        "group_recall_at2": 0,
        "group_recall_at3": 0,
        "recall_at1": 0,
        "recall_at5": 0,
        "recall_at10": 0,
        "recall_at50": 0,
    }
    count = 0

    for item, ranked_names, group_ranking in rankings:
        count += 1
        target = item["target_name"]
        for k in (1, 2, 3):
            metric_hits[f"group_recall_at{k}"] += target in group_ranking[:k]
        for k in (1, 5, 10, 50):
            metric_hits[f"recall_at{k}"] += target in ranked_names[:k]

    return {name: 100.0 * hits / count for name, hits in metric_hits.items()}


@torch.inference_mode()
def evaluate_cirr(
    encoder: QwenEmbeddingEncoder,
    cirr_root: str | Path,
    batch_size: int,
    cache_path: str | Path | None = None,
) -> dict[str, float]:
    encoder.eval()
    classic_dataset = CIRRDataset("val", "classic", cirr_root)
    relative_dataset = CIRRDataset("val", "relative", cirr_root)
    index_features, index_names = load_or_extract_index_features(
        classic_dataset, encoder, batch_size, cache_path
    )
    rankings = rank_queries(
        relative_dataset, encoder, index_features, index_names, batch_size
    )
    return compute_cirr_metrics(rankings)
