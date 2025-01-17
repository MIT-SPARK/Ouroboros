import pathlib
import imageio.v3 as iio
from ouroboros_salad.salad_model import get_salad_model


def resource_dir():
    return pathlib.Path(__file__).absolute().parent.parent / "resources"


if __name__ == "__main__":
    model = get_salad_model()
    img_d = iio.imread(resource_dir() / "arch.jpg")

    out = model(img_d)
