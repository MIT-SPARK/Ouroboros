from ..common.matcher_base import MatcherBase
from ..common.viz2d import plot_images, plot_matches, plot_keypoints
from .lightglue import LightGlue
from .superpoint import SuperPoint
from .utils import load_image, rbd

from typing import List, Tuple
import matplotlib.pyplot as plt
import torch

class LightGlueMatcher(MatcherBase):
    def __init__(self, max_num_keypoints=2048, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def load_image(self, img_path, resize):
        if isinstance(resize, List) or isinstance(resize, Tuple):
            resize = [resize[1], resize[0]]
        return load_image(img_path, resize=resize).to(self.device)

    def execute(self, img0, img1):
        # Extract local features
        feats0 = self.extractor.extract(img0)
        feats1 = self.extractor.extract(img1)

        # Match the features
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

        # Keep the matching keypoints
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return kpts0, kpts1, m_kpts0, m_kpts1

    def plot(self, img0, img1, m_kpts0, m_kpts1, kpts0, kpts1):
        plot_images([img0, img1])
        plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        plot_keypoints([kpts0, kpts1], colors=["blue", "blue"], ps=10)
        plt.show()

    def get_matcher_model(self):
        return self.matcher

    def get_descriptor_model(self):
        return self.extractor

