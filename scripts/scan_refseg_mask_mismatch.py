#!/usr/bin/env python
"""Scan REFER-style segmentation datasets for mask/image size mismatches."""
import argparse
import os
import sys

from pycocotools import mask as mask_utils

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XSAM_PKG_ROOT = os.path.join(ROOT_DIR, "xsam")
for path in (ROOT_DIR, XSAM_PKG_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from xsam.dataset.utils.refer import REFER  # noqa: E402


def decode_shape(segmentation, height, width):
    if isinstance(segmentation, dict):
        mask = mask_utils.decode(segmentation)
        return tuple(mask.squeeze().shape)
    if isinstance(segmentation, list) and len(segmentation) == 0:
        raise ValueError("empty segmentation list")
    if isinstance(segmentation[0], dict):
        shapes = []
        for seg in segmentation:
            mask = mask_utils.decode(seg)
            shapes.append(tuple(mask.squeeze().shape))
        return shapes[0] if len(set(shapes)) == 1 else shapes
    if isinstance(segmentation[0], list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        mask = mask_utils.decode(rles)
        if len(mask.shape) < 3:
            return tuple(mask.squeeze().shape)
        return tuple(mask.shape[:2])
    raise ValueError(f"unsupported segmentation type: {type(segmentation)}")


def scan_dataset(data_root, dataset, split, limit=None, only_mismatch=False):
    refer = REFER(data_root=data_root, dataset=dataset)
    ref_ids = set(refer.getRefIds(split=split)) if split else None

    total = 0
    mismatches = 0
    failures = 0

    for ann_id, ann in refer.Anns.items():
        ref = refer.annToRef.get(ann_id)
        if ref is None:
            continue
        if ref_ids is not None and ref["ref_id"] not in ref_ids:
            continue

        image = refer.Imgs[ann["image_id"]]
        expected_shape = (image["height"], image["width"])
        total += 1

        try:
            got_shape = decode_shape(ann["segmentation"], image["height"], image["width"])
            is_mismatch = got_shape != expected_shape
            if is_mismatch:
                mismatches += 1
                print(
                    "[MISMATCH] "
                    f"image_id={ann['image_id']} ann_id={ann_id} ref_id={ref['ref_id']} "
                    f"split={ref.get('split', '?')} expected={expected_shape} got={got_shape} "
                    f"file_name={image.get('file_name', '?')}"
                )
            elif not only_mismatch:
                print(
                    "[OK] "
                    f"image_id={ann['image_id']} ann_id={ann_id} ref_id={ref['ref_id']} "
                    f"split={ref.get('split', '?')} shape={got_shape}"
                )
        except Exception as e:
            failures += 1
            print(
                "[ERROR] "
                f"image_id={ann['image_id']} ann_id={ann_id} ref_id={ref['ref_id']} "
                f"split={ref.get('split', '?')} expected={expected_shape} error={e}"
            )

        if limit is not None and total >= limit:
            break

    print(
        "\nDone. "
        f"checked={total} mismatches={mismatches} failures={failures} "
        f"dataset={dataset} split={split or 'all'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Scan REFER-style datasets for mask shape mismatches")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root passed to REFER")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. rrsisd")
    parser.add_argument("--split", type=str, default="train", help="Split name, e.g. train/val/test")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N annotations")
    parser.add_argument(
        "--only-mismatch",
        action="store_true",
        help="Only print mismatches and decode errors",
    )
    args = parser.parse_args()

    scan_dataset(
        data_root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        only_mismatch=args.only_mismatch,
    )


if __name__ == "__main__":
    main()