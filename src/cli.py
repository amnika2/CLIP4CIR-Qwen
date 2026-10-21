from argparse import ArgumentParser
from pathlib import Path

from data_utils import DEFAULT_CIRR_ROOT
from qwen_encoder import DEFAULT_MODEL, DEFAULT_QUERY_INSTRUCTION

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_model_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cirr-root", type=Path, default=DEFAULT_CIRR_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)


def add_retrieval_arguments(parser: ArgumentParser, split: str) -> None:
    add_model_arguments(parser)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--index-cache", type=Path, default=PROJECT_ROOT / f"cache/cirr_{split}.pt"
    )
