from lightglue import SuperPoint
import ouroboros as ob
import torch
import numpy as np


class OuroborosSuperPointWrapper:
    def __init__(self, model):
        self.model = model
        self.returns_descriptors = True

    def infer(self, image: ob.SparkImage, pose_hint: ob.VlcPose = None):
        img_float = torch.tensor(
            np.array([image.rgb.transpose() / 255.0]).astype(np.float32)
        ).cuda()
        with torch.no_grad():
            ret = self.model({"image": img_float})

        # Superpoint seems to return (y, x) keypoint coordinates, but our interface assumes keypoints in (x,y)
        return ret["keypoints"].cpu().numpy()[0, :, ::-1], ret[
            "descriptors"
        ].cpu().numpy()[0]


def get_superpoint_model(max_keypoints=1024):
    # Magically loads some weights from torchub
    model = SuperPoint(max_num_keypoints=max_keypoints).eval().cuda()
    return OuroborosSuperPointWrapper(model)
