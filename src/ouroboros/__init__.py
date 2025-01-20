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
from ouroboros.vlc_server.vlc_server import VlcServer

discovered_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg in pkgutil.iter_modules()
    if name.startswith("ouroboros_")
}

# TOOD(nathan) register discovered
