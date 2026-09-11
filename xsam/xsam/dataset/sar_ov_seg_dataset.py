"""
SAR open-vocabulary panoptic dataset for training.

Extends OVSegDataset so prompt synonym augmentation still works,
while keeping SAR category_id==0 (ship) via GenericSegDataset.decode.
"""
import logging

from xsam.dataset.ov_seg_dataset import OVSegDataset
from xsam.utils.logging import print_log


class SarOVSegDataset(OVSegDataset):
    """SAR COCO-panoptic loader for ovseg training."""

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        self.modality = int(kwargs.get("modality", 1))
        if (
            "panoptic" not in self.data_name
            and "pano_ovseg" not in self.data_name
            and "ovseg" not in self.data_name
        ):
            print_log(
                f"SarOVSegDataset: data_name='{self.data_name}' 建议包含 "
                f"'panoptic' / 'pano_ovseg' / 'ovseg'。",
                logger="current",
                level=logging.WARNING,
            )

    def _load_ann_data(self):
        rets = super()._load_ann_data()
        for ret in rets:
            ret["modality"] = self.modality
        print_log(
            f"SarOVSegDataset: loaded {len(rets)} samples, modality={self.modality}, "
            f"data_name={self.data_name}",
            logger="current",
        )
        return rets

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        data_dict.setdefault("modality", self.modality)
        return data_dict
