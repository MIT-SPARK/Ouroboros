"""Test ROS conversions."""

from datetime import datetime

import numpy as np
import numpy.testing as npt
import setup_tests
from server.conversions import (
    spark_image_from_msg,
    spark_image_to_msg,
    vlc_image_from_msg,
    vlc_image_metadata_from_msg,
    vlc_image_metadata_to_msg,
    vlc_image_to_msg,
    vlc_pose_from_msg,
    vlc_pose_to_msg,
)

import ouroboros as ob

assert setup_tests
# TODO(Yun) figure out cleaner way to get around ROS2 and importing


def test_vlc_metadata_conversion():
    vlc_db = ob.VlcDb(3)
    session_id = vlc_db.add_session(0)
    img_stamp = datetime.now()
    img_uuid = vlc_db.add_image(session_id, img_stamp, ob.SparkImage())

    metadata = vlc_db.get_image(img_uuid).metadata

    metadata_msg = vlc_image_metadata_to_msg(metadata)
    metadata_converted = vlc_image_metadata_from_msg(metadata_msg)

    assert metadata_converted.image_uuid == metadata.image_uuid
    assert metadata_converted.session_id == metadata.session_id
    assert metadata_converted.epoch_ns == metadata.epoch_ns


def test_spark_image_conversion():
    height = 10
    width = 10
    rgb_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    depth_image = np.random.uniform(0, 10, (height, width)).astype(np.float32)

    image = ob.SparkImage(rgb=rgb_image, depth=depth_image)
    image_msg = spark_image_to_msg(image)
    image_converted = spark_image_from_msg(image_msg)

    npt.assert_array_equal(image_converted.rgb, image.rgb)
    npt.assert_array_equal(image_converted.depth, image.depth)


def test_vlc_pose_conversion():
    pose = ob.VlcPose(
        time_ns=100, position=np.array([1, 2, 3]), rotation=np.array([1, 0, 0, 0])
    )

    geom_msg = vlc_pose_to_msg(pose)
    pose_converted = vlc_pose_from_msg(geom_msg)

    assert pose_converted.time_ns == pose.time_ns
    npt.assert_array_equal(pose_converted.position, pose.position)
    npt.assert_array_equal(pose_converted.rotation, pose.rotation)


def test_vlc_image_conversion():
    height = 10
    width = 10
    rgb_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    depth_image = np.random.uniform(0, 10, (height, width)).astype(np.float32)
    embedding = np.random.uniform(0, 1, (100,)).astype(np.float32)
    keypoints = np.random.uniform(0, 10, (10, 3)).astype(np.float32)
    descriptors = np.random.uniform(0, 1, (10, 256)).astype(np.float32)

    vlc_db = ob.VlcDb(100)
    session_id = vlc_db.add_session(0)
    img_stamp = datetime.now()
    img_uuid = vlc_db.add_image(
        session_id, img_stamp, ob.SparkImage(rgb=rgb_image, depth=depth_image)
    )
    vlc_db.update_embedding(img_uuid, embedding)
    vlc_db.update_keypoints(img_uuid, keypoints, descriptors)

    vlc_img = vlc_db.get_image(img_uuid)

    vlc_img_msg = vlc_image_to_msg(vlc_img)
    vlc_img_converted = vlc_image_from_msg(vlc_img_msg)
    assert vlc_img_converted.metadata == vlc_img.metadata
