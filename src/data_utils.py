import json
from pathlib import Path

from torch.utils.data import Dataset

DEFAULT_CIRR_ROOT = Path("data/cirr")


class CIRRDataset(Dataset):
    """CIRR metadata dataset that returns image paths and modification text."""

    def __init__(self, split: str, mode: str, root: str | Path = DEFAULT_CIRR_ROOT):
        self.root = Path(root)
        self.split = split
        self.mode = mode

        annotations = self.root / "cirr"
        with (annotations / "image_splits" / f"split.rc2.{split}.json").open() as file:
            self.name_to_relpath = json.load(file)

        self.image_names = list(self.name_to_relpath)
        if mode == "relative":
            with (annotations / "captions" / f"cap.rc2.{split}.json").open() as file:
                self.triplets = json.load(file)

    def image_path(self, image_name: str) -> str:
        return str(self.root / self.name_to_relpath[image_name])

    def __len__(self) -> int:
        return len(self.triplets) if self.mode == "relative" else len(self.image_names)

    def __getitem__(self, index: int) -> dict:
        if self.mode == "classic":
            image_name = self.image_names[index]
            return {"image_name": image_name, "image": self.image_path(image_name)}

        triplet = self.triplets[index]
        reference_name = triplet["reference"]
        item = {
            "reference_name": reference_name,
            "reference_image": self.image_path(reference_name),
            "caption": triplet["caption"],
            "group_members": triplet["img_set"]["members"],
        }

        if self.split in {"train", "val"}:
            target_name = triplet["target_hard"]
            item.update(
                {
                    "target_name": target_name,
                    "target_image": self.image_path(target_name),
                }
            )
        else:
            item["pair_id"] = triplet["pairid"]
        return item
