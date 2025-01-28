from __future__ import annotations
import numpy as np
import torch
import torchvision.transforms as T
from dataclasses import dataclass

import ouroboros as ob
from ouroboros.config import Config, register_config

torch.backends.cudnn.benchmark = True


class SaladModel:
    def __init__(self, config: SaladModelConfig):
        self.embedding_size = config.embedding_size
        self.similarity_metric = "ip"

        if config.model_source == "torchhub":
            model = torch.hub.load(config.model_variant, config.weight_source)
            self.model = model.eval().to("cuda")
        else:
            raise Exception(f"Unknown model source {config.model_source}")

        self.input_transform = get_input_transform((322, 434))

    def infer(self, image: ob.SparkImage, pose_hint: ob.VlcPose = None):
        img_float = torch.tensor((image.rgb.transpose() / 255.0).astype(np.float32))
        with torch.no_grad():
            img = self.input_transform(img_float)
            out = self.model(img.unsqueeze(0).to("cuda")).cpu().numpy()
        return np.squeeze(out)

    @classmethod
    def load(cls, path):
        config = ob.config.Config.load(SaladModelConfig, path)
        return cls(config)


@register_config("place_model", name="Salad", constructor=SaladModel)
@dataclass
class SaladModelConfig(Config):
    embedding_size: int = 8448
    model_source: str = "torchhub"
    model_variant: str = "serizba/salad"
    weight_source: str = "dinov2_salad"


def get_input_transform(image_size=None):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    if image_size:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        return T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
