import pathlib

import imageio.v3 as iio

import ouroboros as ob
from ouroboros_salad.salad_model import get_salad_model


def resource_dir():
    return pathlib.Path(__file__).absolute().parent.parent / "resources"


if __name__ == "__main__":
    model = get_salad_model()
    img_d = iio.imread(resource_dir() / "arch.jpg")

    simg = ob.SparkImage(rgb=img_d)
    out = model.infer(simg)
    print("Salad model returned: ", out)
