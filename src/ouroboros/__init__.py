import importlib
import pkgutil

from ouroboros.vlc_db import (
    KeypointSizeException,
    SparkImage,
    SparkLoopClosure,
    SparkLoopClosureMetadata,
    SparkSession,
    VlcDb,
    VlcImage,
    VlcImageMetadata,
    VlcPose,
)
from ouroboros.vlc_server.vlc_server import VlcServer, VlcServerConfig


def discover_plugins():
    discovered_plugins = {
        name: importlib.import_module(name)
        for finder, name, ispkg in pkgutil.iter_modules()
        if name.startswith("ouroboros_")
    }
    return discovered_plugins


# TOOD(nathan) register discovered
