import os.path as osp
import unittest
from typing import Dict, Tuple

from PIL import Image
from ymir_exc.dataset_convert.ymir2mmseg import convert_rgb_to_label_id


class TestSeg(unittest.TestCase):
    def load_labelmap(self) -> Dict[Tuple, int]:
        label_map_txt = osp.join("tests/seg_images", "labelmap.txt")
        with open(label_map_txt, "r") as fp:
            lines = fp.readlines()

        palette_dict: Dict[Tuple, int] = {}
        for idx, line in enumerate(lines):
            label, rgb = line.split(":")[0:2]
            r, g, b = [int(x) for x in rgb.split(",")]
            palette_dict[(r, g, b)] = idx
        return palette_dict

    def test_image_convert(self):
        gt_img_dict = dict()
        for key in ["color", "instanceIds", "labelIds", "labelTrainIds.png"]:
            gt_img_dict[key] = Image.open(
                osp.join("tests/seg_images", f"munster_000000_000019_gtFine_{key}.png")
            )

        palette_dict = self.load_labelmap()
        gt_img_dict["ymir"] = convert_rgb_to_label_id(
            gt_img_dict["color"], palette_dict
        )
