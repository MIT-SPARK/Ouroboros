import logging
import pathlib
from datetime import datetime

import click
import spark_config as sc
import tqdm
from spark_dataset_interfaces.rosbag_dataloader import RosbagDataLoader

import ouroboros as ob


def _register_camera(server, intrinsics, camera_name):
    conf = ob.PinholeCamera(
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
    )
    return server.register_camera(0, conf, datetime.now(), name=camera_name)


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
@click.option("--verbose", "-v", is_flag=True)
def cli(verbose):
    """Utilities for computing visual loop closures."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = ClickHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)


@cli.command()
@click.argument("bag_path", type=click.Path(exists=True))
@click.argument("rgb_topic")
@click.option("--config-name", "-c", default="salad_server.yaml")
@click.option(
    "--camera-info",
    default=None,
    help="Camera info topic to use (if not specified, will derive from RGB_TOPIC)",
)
@click.option(
    "--depth-topic",
    "-d",
    default=None,
    help="Depth topic to use (will not save keypoint depth if ommitted)",
)
@click.option(
    "--frame-period",
    default=0.5,
    type=float,
    help="Separation between keyframes in seconds",
)
@click.option(
    "--max-frames", "-n", default=None, type=int, help="Stop processing after N frames"
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output database to specific path"
)
@click.option("--name", default=None, help="Session name to use")
@click.option(
    "--append",
    "-a",
    default=None,
    type=click.Path(exists=True),
    help="Load and append to prior database",
)
@click.option("--sync_diff_us", default=1, type=int, help="max diff for topic sync")
def bag(
    bag_path,
    rgb_topic,
    config_name,
    camera_info,
    depth_topic,
    frame_period,
    max_frames,
    output,
    name,
    append,
    sync_diff_us,
):
    """
    Save descriptors and features from a rosbag.

    Optionally loads a prior set of sessions (via `--append`) and adds the
    current session to it.

    Positional Arguments:
        BAG_PATH: Path to rosbag to process
        RGB_TOPIC: Color camera topic to use
    """
    plugins = sc.discover_plugins("ouroboros_")
    logging.info(f"Discovered Plugins: {[x for x in plugins]}")

    bag_path = pathlib.Path(bag_path).expanduser().resolve()
    if camera_info is None:
        rgb_path = pathlib.Path(rgb_topic)
        if rgb_path.stem == "compressed":
            rgb_path = rgb_path.parent

        camera_info = str(rgb_path.parent / "camera_info")

    loader = RosbagDataLoader(
        bag_path,
        rgb_topic,
        camera_info,
        depth_topic=depth_topic,
        threshold_us=sync_diff_us,
    )
    config = ob.VlcServerConfig.load(ob.config_path() / config_name)
    config.strict_keypoint_evaluation = True
    server = ob.VlcServer(config, robot_id=0)
    if append:
        server.load_db(append)

    last_time = None
    min_diff_ns = int(1.0e9 * frame_period)
    with loader:
        if name is None:
            name = bag_path.stem

        num_added = 0
        session_id = _register_camera(server, loader.intrinsics, name)
        for data in tqdm.tqdm(loader):
            time = data.timestamp
            rgb = data.color
            depth = data.depth
            if last_time is not None and time - last_time < min_diff_ns:
                continue

            last_time = time
            img = ob.SparkImage(rgb=rgb, depth=depth)
            server.add_frame(session_id, img, time)
            num_added += 1
            if max_frames is not None and num_added >= max_frames:
                break

    output_path = f"vlc_db_{bag_path.stem}.pkl"
    if output is not None:
        output_path = output
    elif append is not None:
        output_path = append

    server.save_db(output_path)


if __name__ == "__main__":
    cli()
