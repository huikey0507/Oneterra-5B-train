# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
import json
import logging
import os
import os.path as osp
import random
import tempfile

import numpy as np
import torch

from xsam.utils.logging import print_log

from ..utils.constants import DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN
from .base_dataset import BaseDataset
from .utils.catalog import MetadataCatalog
from .utils.coco import COCO
from .utils.mask import decode_mask, is_segmentation_decodable
from .utils.refer import REFER

SPECIAL_TOKENS = [DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN]


class ReferSegDataset(BaseDataset):
    def __init__(
        self,
        *args,
        task_name="refseg",
        dataset=None,
        data_root=None,
        data_split=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            data_path=None,
            dataset=dataset,
            task_name=task_name,
            data_root=data_root,
            data_split=data_split,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        self.dataset = kwargs.get("dataset", None)
        self.data_root = kwargs.get("data_root", None)
        self.data_split = kwargs.get("data_split", None)

    def _set_metadata(self, **kwargs):
        gt_json = kwargs.get("gt_json", None)
        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=gt_json,
            data_name=self.data_name,
            ignore_label=self.ignore_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _convert_to_coco_format(self):
        refer_api = REFER(self.data_root, self.dataset)
        coco_data = {"images": [], "annotations": [], "categories": []}

        # images
        for img_id, img in refer_api.Imgs.items():
            ref = refer_api.imgToRefs.get(img_id, None)
            if ref is None or len(ref) == 0:
                continue  # Skip images without references
            if ref[0]["split"] != self.data_split:
                continue
            # 保持原始 file_name，让 _resolve_image_path 函数处理路径解析
            # 如果 file_name 是绝对路径，_resolve_image_path 会直接使用
            # 如果是相对路径，会与 image_folder 组合
            file_name = img["file_name"]
            coco_data["images"].append(
                {
                    "id": img_id,
                    "file_name": file_name,
                    "height": img["height"],
                    "width": img["width"],
                }
            )

        # annotations
        for ann_id, ann in refer_api.Anns.items():
            assert (isinstance(ann["segmentation"], list) and len(ann["segmentation"]) > 0) or isinstance(
                ann["segmentation"], dict
            )
            ref = refer_api.annToRef.get(ann_id, None)
            # NOTE: one ref may have multiple sentences, but only one annotation.
            # Skip annotations without references
            if not ref:
                continue
            if ref["split"] != self.data_split:
                continue
            
            # Get category_id from annotation or ref
            category_id = ann.get("category_id", None)
            if category_id is None:
                category_id = ref.get("category_id", None)
            if category_id is None:
                # Skip if no category_id found
                continue
            
            cur_ann = {
                "id": ann_id,
                "image_id": ann["image_id"],
                "category_id": category_id,
                "segmentation": ann["segmentation"],
                "area": ann["area"],
                "bbox": ann["bbox"],
                "iscrowd": ann.get("iscrowd", 0),
                "refer_sents": [sent for sent in ref["sentences"]],
            }

            # only add the annotation if it has refer expressions
            coco_data["annotations"].append(cur_ann)

        # categories as placeholder
        for cat_id, cat_name in refer_api.Cats.items():
            coco_data["categories"].append({"id": cat_id, "name": cat_name})

        return coco_data

    @staticmethod
    def _normalize_ann_segmentation(ann):
        segmentation = ann["segmentation"]
        if isinstance(segmentation, list) and len(segmentation) > 0 and isinstance(segmentation[0], dict):
            return dict(ann, segmentation=segmentation[0])
        return ann

    def _filter_valid_anns(self, anns, height, width):
        valid_anns = []
        for ann in anns:
            norm_ann = self._normalize_ann_segmentation(ann)
            if is_segmentation_decodable(norm_ann["segmentation"], height, width):
                valid_anns.append(norm_ann)
            else:
                self.skipped_bad_mask_cnt += 1
                print_log(
                    f"Filtered undecodable mask: image_id={ann.get('image_id', '?')} ann_id={ann.get('id', '?')} "
                    f"({self.data_name})",
                    logger="current",
                    level=logging.WARNING,
                )
        return valid_anns

    def _load_ann_data(self):
        # 固定随机种子，使多进程/多卡构建的数据集一致，避免 DistributedSampler 因长度不一致崩溃
        random.seed(42)
        self.skipped_bad_mask_cnt = 0
        coco_data = self._convert_to_coco_format()
        coco_api = COCO(dataset=coco_data)
        img_ids = sorted(coco_api.getImgIds())

        rets = []
        for img_id in img_ids:
            _img_info = coco_api.loadImgs(img_id)[0]
            ann_ids = coco_api.getAnnIds(imgIds=[img_id])
            anns = coco_api.loadAnns(ann_ids)
            _anns = self._filter_valid_anns(anns, _img_info["height"], _img_info["width"])
            if len(_anns) == 0:
                self.woann_cnt += 1
                continue

            img_info = {
                "file_name": _img_info["file_name"],
                "image_id": _img_info["id"],
                "height": _img_info["height"],
                "width": _img_info["width"],
            }

            # 兼容句子格式：REFER 标准为 dict(sent_id, sent, tokens)，部分数据集可能为 str
            def _sent_to_str(x):
                return x.get("sent", x) if isinstance(x, dict) else str(x)
            ann_sents = [sorted(list(set(_sent_to_str(x) for x in ann.pop("refer_sents")))) for ann in _anns]
            if self.data_split == "train":
                sent_combinations = list(itertools.product(*ann_sents))
                sent_combinations = random.sample(
                    sent_combinations, min(len(sent_combinations), sum(len(x) for x in ann_sents))
                )
                anns = [copy.deepcopy(ann) for ann in _anns]
            else:
                sent_combinations = [sum(ann_sents, [])]
                anns = sum(
                    [[copy.deepcopy(ann) for _ in range(len(ann_sent))] for ann, ann_sent in zip(_anns, ann_sents)], []
                )

            for sent_combination in sent_combinations:
                assert len(sent_combination) == len(anns)
                sampled_anns = copy.deepcopy(anns)
                sampled_sents = list(sent_combination)

                sampled_inds = random.sample(range(len(sampled_sents)), min(len(sampled_sents), self.sample_num))
                sampled_sents = [sampled_sents[i] for i in sampled_inds]
                sampled_anns = [sampled_anns[i] for i in sampled_inds]

                # 若采样结果为 0（如 sample_num=0），跳过该条，避免 _decode_mask 中 torch.stack([]) 报错
                if len(sampled_sents) == 0:
                    continue

                for i, (sampled_sent, sampled_ann) in enumerate(zip(sampled_sents, sampled_anns)):
                    sampled_ann["category_id"] = i
                    print_log(f"Image ID {img_id}: sampled sent '{sampled_sent}' assigned category_id {i}", logger="current")

                if self.data_split != "train":
                    for i, (sampled_sent, sampled_ann) in enumerate(zip(sampled_sents, sampled_anns)):
                        rets.append(
                            {
                                "image_file": _img_info["file_name"],
                                "image_id": _img_info["id"],
                                "image_size": (_img_info["height"], _img_info["width"]),
                                "sampled_sents": [sampled_sent],
                                "annotations": [sampled_ann],
                                "image_info": {**img_info, "sample_id": i, "phrases": [sampled_sent]},
                            }
                        )
                else:
                    rets.append(
                        {
                            "image_file": _img_info["file_name"],
                            "image_id": _img_info["id"],
                            "image_size": (_img_info["height"], _img_info["width"]),
                            "sampled_sents": sampled_sents,
                            "annotations": sampled_anns,
                            "image_info": {**img_info, "phrases": sampled_sents},
                        }
                    )

        if self.data_split != "train":
            base_temp = tempfile.gettempdir()
            cache_dir = osp.join(base_temp, "xsam_cache")
            os.makedirs(cache_dir, exist_ok=True)
            temp_dir = tempfile.mkdtemp(dir=cache_dir)
            print_log(f"Writing {self.data_name} gt_json to {temp_dir}...", logger="current")
            temp_file = osp.join(temp_dir, f"{self.data_name}.json")
            with open(temp_file, "w") as f:
                json.dump(rets, f)
            self._set_metadata(gt_json=temp_file)
        else:
            self._set_metadata()

        if self.skipped_bad_mask_cnt > 0:
            print_log(
                f"Filtered {self.skipped_bad_mask_cnt} annotations with undecodable masks in {self.data_name}.",
                logger="current",
            )

        del coco_data
        return rets

    def _decode_mask(self, data_dict):
        height, width = data_dict["image_size"]
        sampled_anns = data_dict["annotations"]
        if len(sampled_anns) == 0:
            raise ValueError(
                "refer_seg_dataset: annotations is empty for one sample. "
                "This should not happen if _load_ann_data skips empty sampled_sents."
            )
        mask_labels = []
        class_labels = []
        image_id = data_dict.get("image_id", None)
        for idx, ann in enumerate(sampled_anns):
            segmentation = ann["segmentation"]
            try:
                binary_mask = decode_mask(segmentation, height, width)
            except Exception as e:
                seg_type = type(segmentation).__name__
                if isinstance(segmentation, dict):
                    seg_type += f", counts type={type(segmentation.get('counts')).__name__}"
                raise RuntimeError(
                    f"refer_seg_dataset _decode_mask failed for image_id={image_id} ann_idx={idx} "
                    f"(data_name={getattr(self, 'data_name', '?')}): segmentation type={seg_type}. Original: {e}"
                ) from e
            mask_labels.append(binary_mask)
            class_labels.append(ann["category_id"])

        mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in mask_labels])
        class_labels = torch.tensor(np.array(class_labels), dtype=torch.int64)

        data_dict.update(
            {
                "mask_labels": mask_labels,
                "class_labels": class_labels,
            }
        )

        return data_dict
