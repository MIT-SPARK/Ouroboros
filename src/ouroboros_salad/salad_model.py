import numpy as np
import torch
import torchvision.transforms as T

import ouroboros as ob

torch.backends.cudnn.benchmark = True


class SaladModel:
    def __init__(self, model):
        self.embedding_size = 8448
        self.similarity_metric = "ip"
        self.model = model

        self.input_transform = get_input_transform((322, 434))

    def infer(self, image: ob.SparkImage, pose_hint: ob.VlcPose = None):
        img_float = torch.tensor((image.rgb.transpose() / 255.0).astype(np.float32))
        img = self.input_transform(img_float)
        out = self.model(img.unsqueeze(0).to("cuda")).cpu().detach().numpy()
        return np.squeeze(out)


def get_salad_model():
    model = torch.hub.load("serizba/salad", "dinov2_salad")
    model = model.to("cuda")

    return SaladModel(model)


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


# def get_descriptors(model, dataloader, device):
#    descriptors = []
#    with torch.no_grad():
#        with torch.autocast(device_type="cuda", dtype=torch.float16):
#            for batch in tqdm(dataloader, "Calculating descritptors..."):
#                imgs, labels = batch
#                output = model(imgs.to(device)).cpu()
#                descriptors.append(output)
#
#    return torch.cat(descriptors)
