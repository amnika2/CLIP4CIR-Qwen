from collections.abc import Sequence
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_QUERY_INSTRUCTION = (
    "Retrieve the target image that matches the reference image after applying "
    "the user's modification text."
)


class QwenEmbeddingEncoder(torch.nn.Module):
    """Frozen Qwen3-VL-Embedding encoder for CIRR queries and images."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: str = "bfloat16",
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        adapter_path: str | Path | None = None,
    ):
        super().__init__()
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": getattr(torch, dtype)},
        )
        if adapter_path is not None:
            adapter_path = Path(adapter_path).resolve()
            self.model.load_adapter(str(adapter_path))
            adapter_file = adapter_path / "adapter_model.safetensors"
            adapter_stat = adapter_file.stat()
            model_id = (
                f"{model_name}::adapter={adapter_path}"
                f"::{adapter_stat.st_size}:{adapter_stat.st_mtime_ns}"
            )
        else:
            model_id = model_name

        self.model.requires_grad_(False)
        self.model_name = model_id
        self.dtype = dtype
        self.query_instruction = query_instruction
        self.document_instruction = self.model._resolve_prompt(None, None)
        self.eval()

    @torch.inference_mode()
    def encode_queries(
        self,
        images: Sequence[str],
        captions: Sequence[str],
        batch_size: int,
        show_progress: bool = False,
    ) -> torch.Tensor:
        inputs = [
            {"image": image, "text": caption}
            for image, caption in zip(images, captions)
        ]
        return self._encode(inputs, batch_size, self.query_instruction, show_progress)

    @torch.inference_mode()
    def encode_images(
        self,
        images: Sequence[str],
        batch_size: int,
        show_progress: bool = False,
    ) -> torch.Tensor:
        inputs = [{"image": image} for image in images]
        return self._encode(
            inputs, batch_size, self.document_instruction, show_progress
        )

    def _encode(
        self,
        inputs: list[dict],
        batch_size: int,
        prompt: str | None,
        show_progress: bool,
    ) -> torch.Tensor:
        embeddings = self.model.encode(
            inputs,
            prompt=prompt,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings.float().cpu()
