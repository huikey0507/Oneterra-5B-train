"""
SAR panoptic dataset for training.

Inherits GenericSegDataset (NOT PanoSegDataset):
- Optical PanoSegDataset skips category_id==0 as background.
- SAR uses category_id==0 for ship (thing), which must be kept.
- Mask encoding is standard COCO panoptic rgb2id = R + 256*G + 256^2*B;
  category comes from segments_info lookup, same as optical.
"""
import logging

from xsam.dataset.generic_seg_dataset import GenericSegDataset
from xsam.utils.logging import print_log


class SarPanoSegDataset(GenericSegDataset):
    """SAR COCO-panoptic loader for genseg / panoptic training."""

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        # 1 = SAR, 0 = optical; used later by modality gate / adapters.
        self.modality = int(kwargs.get("modality", 1))
        if "panoptic" not in self.data_name and "pano" not in self.data_name:
            print_log(
                f"SarPanoSegDataset: data_name='{self.data_name}' 建议包含 "
                f"'panoptic'（或 'pano'）以走全景加载/解码分支。",
                logger="current",
                level=logging.WARNING,
            )

    def _load_ann_data(self):
        rets = super()._load_ann_data()
        for ret in rets:
            ret["modality"] = self.modality
            # Parent sets seg_map via image_file .jpg -> .png, which matches PanSAR2.
        print_log(
            f"SarPanoSegDataset: loaded {len(rets)} samples, modality={self.modality}, "
            f"data_name={self.data_name}",
            logger="current",
        )
        return rets

    def _decode_mask(self, data_dict):
        """Keep all non-crowd segments including category_id==0 (ship)."""
        return super()._decode_mask(data_dict)

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        # Ensure modality survives deepcopy / skip-retry paths.
        data_dict.setdefault("modality", self.modality)
        return data_dict
