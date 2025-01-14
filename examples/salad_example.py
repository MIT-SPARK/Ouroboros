import torch
import torchvision.transforms as T
import numpy as np

from importlib_resources import files
import vlc_resources
import imageio.v2 as imageio
import tqdm

torch.backends.cudnn.benchmark = True


def get_salad_model():
    model = torch.hub.load("serizba/salad", "dinov2_salad")
    model = model.to("cuda")

    trans = input_transform((322, 322))

    def call_model(img_in):
        #img_float = torch.tensor((img_in.transpose() / 255.0).astype(np.float32))
        img_float = torch.tensor((img_in.transpose()).astype(np.float32))
        img = trans(img_float)
        out = model(img.unsqueeze(0).to("cuda")).cpu().detach().numpy()
        return np.squeeze(out)

    return call_model


def input_transform(image_size=None):
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


def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for batch in tqdm(dataloader, "Calculating descritptors..."):
                imgs, labels = batch
                output = model(imgs.to(device)).cpu()
                descriptors.append(output)

    return torch.cat(descriptors)


if __name__ == "__main__":
    model = get_salad_model()
    fn_d = files(vlc_resources).joinpath("arch.jpg")
    img_d = imageio.imread(fn_d)

    out = model(img_d)
