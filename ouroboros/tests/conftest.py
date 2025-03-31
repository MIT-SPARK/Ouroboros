"""Fixtures for unit tests."""

import pathlib

import imageio.v3 as iio
import pytest


def _resource_dir():
    return pathlib.Path(__file__).absolute().parent.parent.parent / "resources"


@pytest.fixture()
def resource_dir() -> pathlib.Path:
    """Get a path to the resource directory for tests."""
    return _resource_dir()


@pytest.fixture()
def image_factory():
    """Get images from resource directory."""
    resource_path = _resource_dir()

    def factory(name):
        """Get image."""
        img_path = resource_path / name
        img = iio.imread(img_path)
        assert img is not None
        return img

    return factory
