import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType
from sentence_transformers.util import batch_to_device

from qwen_encoder import DEFAULT_MODEL, DEFAULT_QUERY_INSTRUCTION, QwenEmbeddingEncoder

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class QwenLoraEncoder(QwenEmbeddingEncoder):
    """Qwen3-VL embedding model with language-attention LoRA adapters."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: str = "bfloat16",
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            dtype=dtype,
            query_instruction=query_instruction,
        )
        self.model.add_adapter(
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=LORA_TARGET_MODULES,
                bias="none",
            )
        )

        if gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    @property
    def transformer(self):
        return self.model[0].auto_model

    def forward_queries(
        self,
        images: list[str],
        captions: list[str],
    ) -> torch.Tensor:
        inputs = [
            {"image": image, "text": caption}
            for image, caption in zip(images, captions)
        ]
        return self._forward(inputs, self.query_instruction)

    def forward_images(self, images: list[str]) -> torch.Tensor:
        inputs = [{"image": image} for image in images]
        return self._forward(inputs, self.document_instruction)

    def _forward(self, inputs: list[dict], prompt: str | None) -> torch.Tensor:
        features = self.model.preprocess(inputs, prompt=prompt)
        features = batch_to_device(features, self.model.device)
        embeddings = self.model(features)["sentence_embedding"]
        return F.normalize(embeddings.float(), dim=-1)

    def save_adapter(self, output_dir: str | Path, metadata: dict) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(output_dir)
        with (output_dir / "training_metadata.json").open("w") as file:
            json.dump(metadata, file, indent=2)

    def trainable_parameter_counts(self) -> tuple[int, int]:
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        return trainable, total
