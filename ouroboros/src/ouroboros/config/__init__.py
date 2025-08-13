import pathlib


def config_path():
    """Get base path to configs."""
    return pathlib.Path(__file__).absolute().parent
