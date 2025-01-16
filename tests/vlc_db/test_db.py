import time
from datetime import datetime

import numpy as np
import pytest

from vlc_db.spark_image import SparkImage
from vlc_db.spark_loop_closure import SparkLoopClosure
from vlc_db.vlc_db import VlcDb


def test_add_image(image_factory):
    vlc_db = VlcDb(3)
    session_id = vlc_db.add_session(0)

    img_a = image_factory("arch.jpg")
    img_b = image_factory("left_img_0.png")

    key_a = vlc_db.add_image(session_id, datetime.now(), SparkImage(rgb=img_a))
    key_b = vlc_db.add_image(session_id, time.time() * 1e9, SparkImage(rgb=img_b))

    assert key_a != key_b, "Image UUIDs not unique"

    vlc_img_a = vlc_db.get_image(key_a)
    assert key_a == vlc_img_a.metadata.image_uuid, "Inconsistent image_uuid"
    assert session_id == vlc_img_a.metadata.session_id, "Inconsistent session_id"

    vlc_img_b = vlc_db.get_image(key_b)

    assert vlc_img_b.metadata.epoch_ns > vlc_img_a.metadata.epoch_ns

    with pytest.raises(KeyError):
        vlc_db.get_image(0)


def test_iterate_image():
    vlc_db = VlcDb(3)
    session_id = vlc_db.add_session(0)
    for _ in range(10):
        t_rand = int(np.random.random() * 1e9)
        vlc_db.add_image(session_id, t_rand, None)

    t_last = -np.inf
    for img in vlc_db.iterate_images():
        t = img.metadata.epoch_ns
        assert t >= t_last, "iterate_images epoch_ns not ascending"
        t_last = t


def test_get_image_keys():
    vlc_db = VlcDb(3)
    session_id = vlc_db.add_session(0)
    for _ in range(10):
        t_rand = int(np.random.random() * 1e9)
        vlc_db.add_image(session_id, t_rand, None)

    t_last = -np.inf
    for k in vlc_db.get_image_keys():
        t = vlc_db.get_image(k).metadata.epoch_ns
        assert t >= t_last, "get_image_keys epoch_ns not ascending"
        t_last = t


def test_query_embedding():
    vlc_db = VlcDb(3)
    session_id = vlc_db.add_session(0)

    a_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(a_id, np.array([0, 0, 1]))

    b_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(b_id, np.array([0, 1, 0]))

    c_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(c_id, np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)]))

    d_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(d_id, np.array([0, 0, 1]))

    imgs, dists = vlc_db.query_embeddings(np.array([[0, 0.2, 0.8]]), 1)
    assert len(imgs) == 1, "number of results differs from number of query vectors"
    assert (
        len(imgs[0]) == 1
    ), "query_embeddings returned wrong number of closest matches"
    assert imgs[0][0].metadata.image_uuid in [a_id, d_id]

    imgs, dists = vlc_db.query_embeddings(np.array([[0, 0.2, 0.8]]), 2)
    assert len(imgs[0]) == 2, "query_embeddings returned wrong number of matches"
    match_ids = []
    for img in imgs[0]:
        match_ids.append(img.metadata.image_uuid)
        assert img.metadata.image_uuid in [
            a_id,
            d_id,
        ], "Query matched to vector other than k closest"

    for img_id in [a_id, d_id]:
        assert img_id in match_ids, "Query didn't match to all of k closest"

    imgs, dists = vlc_db.query_embeddings(np.array([[0, 0.2, 0.8]]), 3)
    assert len(imgs[0]) == 3, "query_embeddings returned wrong number of matches"
    match_ids = []
    for img in imgs[0]:
        match_ids.append(img.metadata.image_uuid)
        assert img.metadata.image_uuid in [
            a_id,
            c_id,
            d_id,
        ], "Query matched to vector other than k closest"

    for img_id in [a_id, c_id, d_id]:
        assert img_id in match_ids, "Query didn't match to all of k closest"

    imgs, dists = vlc_db.query_embeddings(np.array([[0, 1, 0], [0, 0.2, 0.2]]), 1)
    assert len(imgs) == 2, "number of results different from number of query vectors"
    assert len(imgs[0]) == 1, "returned wrong number of matches for vector"
    assert (
        imgs[0][0].metadata.image_uuid == b_id
    ), "Did not match correct image embedding"
    assert (
        imgs[1][0].metadata.image_uuid == c_id
    ), "Did not match correct image embedding"


def test_query_embedding_custom_similarity():
    vlc_db = VlcDb(3)
    session_id = vlc_db.add_session(0)

    a_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(a_id, np.array([0, 0, 1]))

    b_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(b_id, np.array([0, 2, 0]))

    c_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(c_id, np.array([3, 0, 0]))

    def radius_metric(x, y):
        return -abs(np.linalg.norm(x) - np.linalg.norm(y))

    imgs, dists = vlc_db.query_embeddings(
        np.array([[2, 0, 0]]), 1, similarity_metric=radius_metric
    )

    assert (
        imgs[0][0].metadata.image_uuid == b_id
    ), "custom similarity matched wrong vector"


def test_update_keypoints():
    vlc_db = VlcDb(3)
    session_id = vlc_db.add_session(0)

    a_id = vlc_db.add_image(session_id, datetime.now(), None)
    b_id = vlc_db.add_image(session_id, datetime.now(), None)

    kps = np.random.random((10, 2))
    vlc_db.update_keypoints(a_id, kps)
    keypoints, descriptors = vlc_db.get_keypoints(a_id)
    assert np.allclose(keypoints, kps)
    assert descriptors is None

    descriptors_in = np.random.random(kps.shape)
    vlc_db.update_keypoints(b_id, kps, descriptors=descriptors_in)
    keypoints, descriptors = vlc_db.get_keypoints(b_id)
    assert np.allclose(keypoints, kps)
    assert np.allclose(descriptors, descriptors_in)


def test_add_session():
    vlc_db = VlcDb(3)
    robot_id = 0
    sensor_id = 42
    session_id = vlc_db.add_session(robot_id, sensor_id, "session_name")

    session = vlc_db.get_session(session_id)

    assert session.robot_id == 0
    assert session.name == "session_name"
    assert session.sensor_id == 42


def test_insert_session():
    vlc_db = VlcDb(3)
    session_id = 1234
    robot_id = 0
    sensor_id = 42
    vlc_db.insert_session(
        session_id, datetime.now(), robot_id, sensor_id, "session_name"
    )

    session = vlc_db.get_session(session_id)

    assert session.robot_id == 0
    assert session.name == "session_name"
    assert session.sensor_id == 42


def test_add_lc():
    vlc_db = VlcDb(3)
    session_id = vlc_db.add_session(0)

    computed_ts = datetime.now()
    loop_closure = SparkLoopClosure(
        from_image_uuid=0,
        to_image_uuid=1,
        f_T_t=np.eye(4),
        quality=1,
    )

    lc_uuid = vlc_db.add_lc(loop_closure, session_id, creation_time=computed_ts)
    lc = vlc_db.get_lc(lc_uuid)
    assert lc.from_image_uuid == 0
    assert lc.to_image_uuid == 1
    assert lc.f_T_t.shape == (4, 4)
    assert lc.metadata.lc_uuid == lc_uuid
    assert lc.metadata.session_uuid == session_id


def test_iterate_lcs():
    vlc_db = VlcDb(3)
    session_id = vlc_db.add_session(0)

    computed_ts = datetime.now()
    loop_closure = SparkLoopClosure(
        from_image_uuid=0,
        to_image_uuid=1,
        f_T_t=np.eye(4),
        quality=1,
    )
    lc_uuid_1 = vlc_db.add_lc(loop_closure, session_id, creation_time=computed_ts)
    computed_ts_2 = datetime.now()
    loop_closure = SparkLoopClosure(
        from_image_uuid=3,
        to_image_uuid=4,
        f_T_t=np.eye(4),
        quality=1,
    )
    lc_uuid_2 = vlc_db.add_lc(loop_closure, session_id, creation_time=computed_ts_2)

    uuids = []
    for lc in vlc_db.iterate_lcs():
        uuids.append(lc.metadata.lc_uuid)
        assert lc.metadata.lc_uuid in [lc_uuid_1, lc_uuid_2]

    for u in [lc_uuid_1, lc_uuid_2]:
        assert u in uuids
