import logging
import pathlib
from datetime import datetime

import click
import spark_config as sc
import tqdm
from spark_dataset_interfaces.rosbag_dataloader import RosbagDataLoader

import ouroboros as ob


def _register_camera(server, intrinsics):
    conf = ob.PinholeCamera(
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
    )
    return server.register_camera(0, conf, datetime.now())


class ClickHandler(logging.Handler):
    """Logging handler to color output using click."""


    def emit(self, record):
        """Send log record to console with appropriate coloring."""
        msg = self.format(record)


        if record.levelno <= logging.DEBUG:
            click.secho(msg, fg="green")
            return


        if record.levelno <= logging.INFO:
            click.echo(msg)
            return


        if record.levelno <= logging.WARNING:
            click.secho(msg, fg="yellow", err=True)
            return


        click.secho(msg, fg="red", err=True)

@click.group()
def cli():
    """Utilities for computing visual loop closures."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = ClickHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)


@cli.command()
@click.argument("bag_path", type=click.Path(exists=True))
@click.argument("rgb_topic")
@click.option("--config-name", "-c", default="salad_server.yaml")
@click.option("--camera-info", default=None)
@click.option("--depth-topic", default=None)
@click.option("--frame-period", default=0.5, type=float)
@click.option("--max-frame", "-n", default=None, type=int)
def bag(
    bag_path, rgb_topic, config_name, camera_info, depth_topic, frame_period, max_frame
):
    """Save descriptors and features from a rosbag."""
    plugins = sc.discover_plugins("ouroboros_")
    logging.info(f"Discovered Plugins: {[x for x in plugins]}")

    bag_path = pathlib.Path(bag_path).expanduser().resolve()
    if camera_info is None:
        rgb_path = pathlib.Path(rgb_topic)
        if rgb_path.stem == "compressed":
            rgb_path = rgb_path.parent

        camera_info = str(rgb_path.parent / "camera_info")

    loader = RosbagDataLoader(bag_path, rgb_topic, camera_info, depth_topic=depth_topic)
    config = ob.VlcServerConfig.load(ob.config_path() / config_name)
    server = ob.VlcServer(config, robot_id=0)

    min_diff_ns = int(1.0e9 * frame_period)
    last_time = None
    with loader:
        session_id = _register_camera(server, loader.intrinsics)
        for idx, data in tqdm.tqdm(enumerate(loader)):
            time = data.timestamp
            rgb = data.color
            depth = data.depth
            if max_frame is not None and idx > max_frame:
                break

            if last_time is not None and time - last_time > min_diff_ns:
                continue

            last_time = time
            img = ob.SparkImage(rgb=rgb, depth=depth)
            server.add_frame(session_id, img, time)


if __name__ == "__main__":
    cli()
