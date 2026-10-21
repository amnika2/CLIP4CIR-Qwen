import json
from argparse import ArgumentParser

from cli import add_retrieval_arguments
from qwen_encoder import QwenEmbeddingEncoder
from retrieval import evaluate_cirr


def evaluate(args) -> dict[str, float]:
    encoder = QwenEmbeddingEncoder(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        query_instruction=args.query_instruction,
        adapter_path=args.adapter,
    )
    return evaluate_cirr(encoder, args.cirr_root, args.batch_size, args.index_cache)


def parse_args():
    parser = ArgumentParser(description="Evaluate Qwen3-VL-Embedding on CIRR")
    add_retrieval_arguments(parser, "val")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
