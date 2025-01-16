from importlib_resources import files
from ouroboros_superglue.superglue_model import get_superglue_model
import vlc_resources
import imageio.v2 as imageio

if __name__ == "__main__":
    model = get_superglue_model()
    fn_d = files(vlc_resources).joinpath("arch.jpg")
    img_d = imageio.imread(fn_d)

    out = model(img_d)
