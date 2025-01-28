import ouroboros as ob


class OuroborosGtMatches:
    def __init__(self, model):
        self.model = model
        self.returns_descriptors = True

    def infer(
        self, image0: ob.VlcImage, image1: ob.VlcImage, pose_hint: ob.VlcPose = None
    ):
        return None


def get_gt_match_model():
    return OuroborosGtMatches()
