import importlib
import logging
import pkgutil

from ouroboros.config import *
from ouroboros.pose_recovery import *
from ouroboros.vlc_db import (
    KeypointSizeException,
    PinholeCamera,
    SparkImage,
    SparkLoopClosure,
    SparkLoopClosureMetadata,
    SparkSession,
    VlcDb,
    VlcImage,
    VlcImageMetadata,
    VlcPose,
    invert_pose,
    pose_from_quat_trans,
)
from ouroboros.vlc_server.vlc_server import VlcServer, VlcServerConfig


def discover_plugins():
    def _try_load(name):
        try:
            return importlib.import_module(name)
        except ImportError as e:
            logging.getLogger("ouroboros").warning(
                f"Unable to load plugin '{name}': {e}"
            )
            return None

    discovered_plugins = {
        name: _try_load(name)
        for finder, name, ispkg in pkgutil.iter_modules()
        if name.startswith("ouroboros_")
    }
    discovered_plugins = {k: v for k, v in discovered_plugins.items() if v is not None}
    return discovered_plugins


# TOOD(nathan) register discovered
