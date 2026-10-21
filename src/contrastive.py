import torch
import torch.nn.functional as F


def contrastive_loss(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
    logit_scale: float = 100.0,
) -> torch.Tensor:
    """In-batch query-to-target loss for L2-normalized embeddings."""
    logits = logit_scale * query_features @ target_features.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
