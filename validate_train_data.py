#!/usr/bin/env python3
"""校验 xsam_v3_ultimate_4xA40 混合训练用到的数据文件是否齐全。

用法:
  python validate_train_data.py                 # 检查 V3 全部数据集（汇总表）
  python validate_train_data.py --verbose       # 打印全部缺失路径
  python validate_train_data.py --dataset nwpu  # 只检查名称包含 nwpu 的数据集

说明:
  脚本会检查配置里每个 JSON/目录中的每一条样本，--show 只控制「缺失路径打印几条」，
  不影响检查范围。默认只输出汇总；加 --verbose 才列出全部缺图路径。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

# 与 xsam_v3_ultimate_4xA40.py 保持一致
BASE_ROOT = "/mnt_llm_A100_V1/"
DATA_DIR = BASE_ROOT + "shui/LAE/OneTerra-train/datas/"
ONETERRA = BASE_ROOT + "shui/oneterra_data/"
YANGSEN = BASE_ROOT + "yangsen/datasets/"
FITRS_JSON = ONETERRA + "imgconv/FIT-RS/raw_data/train_data_of_each_individual_task/"
FITRS_IMG = ONETERRA + "imgconv/FIT-RS/raw_data/imgv2_split_512_100_vaild"
OPTICAL = ONETERRA + "imgconv/image_caption/"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def resolve_image_path(image_folder: str, image_file: str) -> str:
    """与 xsam.dataset.base_dataset._resolve_image_path 行为一致。"""
    if os.path.isabs(image_file):
        if os.path.isfile(image_file):
            return image_file
        image_file = os.path.basename(image_file)

    image_path = os.path.join(image_folder, image_file)
    if os.path.isfile(image_path):
        return image_path

    base, ext = os.path.splitext(image_file)
    parts = base.split("_")
    if len(parts) >= 2 and parts[0] == parts[1]:
        alt = os.path.join(image_folder, "_".join(parts[1:]) + ext)
        if os.path.isfile(alt):
            return alt
    return image_path


@dataclass
class CheckResult:
    name: str
    task: str
    json_path: str = ""
    total: int = 0
    missing: List[str] = field(default_factory=list)
    extra_notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.total >= 0 and len(self.missing) == 0 and not self.extra_notes

    @property
    def missing_count(self) -> int:
        return len(self.missing)


def check_imgconv_list(name: str, json_path: str, image_folder: str) -> CheckResult:
    res = CheckResult(name=name, task="imgconv", json_path=json_path)
    if not os.path.isfile(json_path):
        res.extra_notes.append(f"JSON 不存在: {json_path}")
        return res
    if not os.path.isdir(image_folder):
        res.extra_notes.append(f"image_folder 不存在: {image_folder}")
        return res

    data = json.load(open(json_path, encoding="utf-8"))
    if not isinstance(data, list):
        res.extra_notes.append("期望 list 格式 (ImgConv JSON)")
        return res

    res.total = len(data)
    for item in data:
        rel = item.get("image")
        if not rel:
            res.missing.append("<empty image field>")
            continue
        path = resolve_image_path(image_folder, rel)
        if not os.path.isfile(path):
            res.missing.append(path)
    return res


def check_pano(name: str, json_path: str, image_folder: str, panseg_folder: str) -> CheckResult:
    res = CheckResult(name=name, task="pano", json_path=json_path)
    if not os.path.isfile(json_path):
        res.extra_notes.append(f"JSON 不存在: {json_path}")
        return res

    data = json.load(open(json_path, encoding="utf-8"))
    images = data.get("images", [])
    res.total = len(images)

    for img in images:
        fn = img["file_name"]
        img_path = resolve_image_path(image_folder, fn)
        if not os.path.isfile(img_path):
            res.missing.append(img_path)

        seg_name = fn.replace(".jpg", ".png")
        seg_path = os.path.join(panseg_folder, seg_name)
        if not os.path.isfile(seg_path):
            res.missing.append(seg_path)
    return res


def check_refseg(name: str, data_root: str, image_folder: str, split: str = "train") -> CheckResult:
    inst = os.path.join(data_root, "instances.json")
    res = CheckResult(name=name, task="refseg", json_path=inst)
    if not os.path.isfile(inst):
        res.extra_notes.append(f"instances.json 不存在: {inst}")
        return res

    data = json.load(open(inst, encoding="utf-8"))
    images = data.get("images", [])
    res.total = len(images)

    # 若存在 refs.p，只检查 train split 用到的 image_id
    img_ids = None
    for ref_name in ("refs(unc).p", "refs(umd).p", "refs(train).p"):
        ref_p = os.path.join(data_root, ref_name)
        if os.path.isfile(ref_p):
            import pickle

            refs = pickle.load(open(ref_p, "rb"))
            img_ids = {r["image_id"] for r in refs if r.get("split") == split}
            break

    for img in images:
        if img_ids is not None and img["id"] not in img_ids:
            continue
        path = resolve_image_path(image_folder, img["file_name"])
        if not os.path.isfile(path):
            res.missing.append(path)
    return res


def check_reaseg_dir(name: str, image_folder: str) -> CheckResult:
    res = CheckResult(name=name, task="reaseg", json_path=image_folder)
    if not os.path.isdir(image_folder):
        res.extra_notes.append(f"目录不存在: {image_folder}")
        return res

    names = sorted(
        f
        for f in os.listdir(image_folder)
        if f.lower().endswith(IMAGE_EXTS) and not f.startswith(".")
    )
    res.total = len(names)
    for img_name in names:
        img_path = os.path.join(image_folder, img_name)
        if not os.path.isfile(img_path):
            res.missing.append(img_path)
            continue
        stem = os.path.splitext(img_name)[0]
        json_path = os.path.join(image_folder, stem + ".json")
        if not os.path.isfile(json_path):
            res.missing.append(json_path)
    return res


def check_reaseg_explain(name: str, explain_path: str, image_folder: str) -> CheckResult:
    res = CheckResult(name=name, task="reaseg_explain", json_path=explain_path)
    if not os.path.isfile(explain_path):
        res.extra_notes.append(f"explain JSON 不存在: {explain_path}")
        return res
    data = json.load(open(explain_path, encoding="utf-8"))
    res.total = len(data)
    for item in data:
        img_name = item.get("image", "")
        path = os.path.join(image_folder, img_name)
        if img_name and not os.path.isfile(path):
            res.missing.append(path)
    return res


# V3 combined_train_dataset 中的全部数据源
V3_DATASETS: Sequence[Tuple[str, Callable[[], CheckResult]]] = [
    (
        "geochat_imgconv",
        lambda: check_imgconv_list(
            "geochat_imgconv",
            DATA_DIR + "img_conv_data/geochat/geochat_mini_30k_PRO.json",
            DATA_DIR + "img_conv_data/geochat/images",
        ),
    ),
    (
        "sar_total_imgconv",
        lambda: check_imgconv_list(
            "sar_total_imgconv",
            YANGSEN + "sar_total/sft/train_mini_30k.json",
            YANGSEN,
        ),
    ),
    (
        "ucm_captions_imgconv",
        lambda: check_imgconv_list(
            "ucm_captions_imgconv",
            OPTICAL + "UCM-Captions/dataset_qwenvl.json",
            OPTICAL + "UCM-Captions/imgs",
        ),
    ),
    (
        "nwpu_captions_imgconv",
        lambda: check_imgconv_list(
            "nwpu_captions_imgconv",
            OPTICAL + "NWPU-Captions/dataset_nwpu_qwenvl_cleaned.json",
            OPTICAL + "NWPU-Captions/NWPU_images",
        ),
    ),
    (
        "rsvqa_lr_imgconv",
        lambda: check_imgconv_list(
            "rsvqa_lr_imgconv",
            ONETERRA + "imgconv/VQA/RSVQA-LR/train_cleaned.json",
            ONETERRA + "imgconv/VQA/RSVQA-LR/Images_LR",
        ),
    ),
    *[
        (
            ds_name,
            lambda p=json_file, n=ds_name: check_imgconv_list(n, FITRS_JSON + p, FITRS_IMG),
        )
        for ds_name, json_file in [
            ("fitrs_complexcompre", "complexcompre_mini_30k.json"),
            ("fitrs_imagecaption", "imagecaption_mini_20k.json"),
            ("fitrs_imageclassification", "imageclassification_mini_20k.json"),
            ("fitrs_multiturn", "multiturn_mini_20k.json"),
            ("fitrs_regioncaption", "regioncaption_mini_20k.json"),
            ("fitrs_vqa", "vqa_mini_30k.json"),
        ]
    ],
    (
        "pano_ovseg",
        lambda: check_pano(
            "pano_ovseg",
            DATA_DIR + "pano/annotations_train.json",
            DATA_DIR + "pano/train/images",
            DATA_DIR + "pano/train/panoptic_labels",
        ),
    ),
    (
        "pano_genseg",
        lambda: check_pano(
            "pano_genseg",
            DATA_DIR + "pano/annotations_train.json",
            DATA_DIR + "pano/train/images",
            DATA_DIR + "pano/train/panoptic_labels",
        ),
    ),
    (
        "remotesam_refseg",
        lambda: check_refseg(
            "remotesam_refseg",
            DATA_DIR + "ref_seg_data/remotesam",
            DATA_DIR + "ref_seg_data/images/remotesam_images",
        ),
    ),
    (
        "fast_refseg",
        lambda: check_refseg(
            "fast_refseg",
            ONETERRA + "refseg/FAST/fast",
            ONETERRA + "refseg/FAST/images",
        ),
    ),
    (
        "earthreason_reaseg",
        lambda: check_reaseg_dir(
            "earthreason_reaseg",
            ONETERRA + "reasonseg/EarthReason_convert/train",
        ),
    ),
    (
        "earthreason_explain",
        lambda: check_reaseg_explain(
            "earthreason_explain",
            ONETERRA + "reasonseg/EarthReason_convert/explanatory/train.json",
            ONETERRA + "reasonseg/EarthReason_convert/train",
        ),
    ),
    (
        "diy1_reaseg",
        lambda: check_reaseg_dir(
            "diy1_reaseg",
            ONETERRA + "reasonseg/diy1/train",
        ),
    ),
    (
        "risbench_refseg",
        lambda: check_refseg(
            "risbench_refseg",
            ONETERRA + "refseg/RISBench/risbench",
            ONETERRA + "refseg/RISBench/RISBench_dataset/img_rgb",
        ),
    ),
]


def run_checks(filter_name: Optional[str] = None) -> List[CheckResult]:
    results = []
    for name, fn in V3_DATASETS:
        if filter_name and filter_name.lower() not in name.lower():
            continue
        print(f"检查 {name} ...", flush=True)
        results.append(fn())
    return results


def print_report(results: List[CheckResult], show: int) -> int:
    print("\n" + "=" * 88)
    print(f"{'数据集':<28} {'任务':<14} {'总数':>8} {'缺失':>8}  状态")
    print("-" * 88)

    total_missing = 0
    failed = 0
    for r in results:
        if r.extra_notes:
            status = "ERROR"
            failed += 1
        elif r.missing_count:
            status = "FAIL"
            failed += 1
            total_missing += r.missing_count
        else:
            status = "OK"

        print(
            f"{r.name:<28} {r.task:<14} {r.total:>8} {r.missing_count:>8}  {status}"
        )
        for note in r.extra_notes:
            print(f"  ! {note}")
        if r.missing_count and show != 0:
            limit = r.missing_count if show < 0 else min(show, r.missing_count)
            for path in r.missing[:limit]:
                print(f"  - {path}")
            if show >= 0 and r.missing_count > show:
                print(f"  ... 还有 {r.missing_count - show} 条（用 --verbose 查看全部）")

    print("=" * 88)
    print(f"合计: {len(results)} 个数据源, 缺失文件 {total_missing} 个, 异常数据集 {failed} 个")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="校验 V3 混合训练数据完整性（检查每一条样本）")
    parser.add_argument("--dataset", type=str, default=None, help="只检查名称包含该子串的数据集")
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        help="每个数据集最多展示多少条缺失路径，0=仅汇总（默认），-1=全部列出",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="等价于 --show -1，打印所有缺失路径",
    )
    args = parser.parse_args()
    show = -1 if args.verbose else args.show

    results = run_checks(args.dataset)
    exit_code = print_report(results, args.show)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
