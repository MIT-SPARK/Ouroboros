import time
from datetime import datetime

import numpy as np
import pytest

import ouroboros as ob


def test_add_image(image_factory):
    vlc_db = ob.VlcDb(3)
    session_id = vlc_db.add_session(0)

    img_a = image_factory("arch.jpg")
    img_b = image_factory("left_img_0.png")

    key_a = vlc_db.add_image(session_id, datetime.now(), ob.SparkImage(rgb=img_a))
    key_b = vlc_db.add_image(session_id, time.time() * 1e9, ob.SparkImage(rgb=img_b))

    assert key_a != key_b, "Image UUIDs not unique"

    vlc_img_a = vlc_db.get_image(key_a)
    assert key_a == vlc_img_a.metadata.image_uuid, "Inconsistent image_uuid"
    assert session_id == vlc_img_a.metadata.session_id, "Inconsistent session_id"

    vlc_img_b = vlc_db.get_image(key_b)

    assert vlc_img_b.metadata.epoch_ns > vlc_img_a.metadata.epoch_ns

    with pytest.raises(KeyError):
        vlc_db.get_image(0)


def test_iterate_image():
    vlc_db = ob.VlcDb(3)
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
    vlc_db = ob.VlcDb(3)
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
    vlc_db = ob.VlcDb(3)
    session_id = vlc_db.add_session(0)

    a_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(a_id, np.array([0, 0, 1]))

    b_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(b_id, np.array([0, 1, 0]))

    c_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(c_id, np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)]))

    d_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(d_id, np.array([0, 0, 1]))

    imgs, sims = vlc_db.query_embeddings(np.array([0, 0.2, 0.8]), 1)
    assert len(imgs) == 1, "query_embeddings returned wrong number of closest matches"
    assert imgs[0].metadata.image_uuid in [a_id, d_id]

    imgs, sims = vlc_db.query_embeddings(np.array([0, 0.2, 0.8]), 2)
    assert len(imgs) == 2, "query_embeddings returned wrong number of matches"
    assert set((a_id, d_id)) == set([img.metadata.image_uuid for img in imgs]), (
        "Query did not match k closest"
    )

    imgs, sims = vlc_db.query_embeddings(np.array([0, 0.2, 0.8]), 3)
    assert len(imgs) == 3, "query_embeddings returned wrong number of matches"

    assert set((a_id, c_id, d_id)) == set([img.metadata.image_uuid for img in imgs]), (
        "Query did not match k closest"
    )


def test_query_embedding_custom_similarity():
    vlc_db = ob.VlcDb(3)
    session_id = vlc_db.add_session(0)

    a_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(a_id, np.array([0, 0, 1]))

    b_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(b_id, np.array([0, 2, 0]))

    c_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(c_id, np.array([3, 0, 0]))

    def radius_metric(x, y):
        return -abs(np.linalg.norm(x) - np.linalg.norm(y))

    imgs, sims = vlc_db.query_embeddings(
        np.array([2, 0, 0]), 1, similarity_metric=radius_metric
    )

    assert imgs[0].metadata.image_uuid == b_id, "custom similarity matched wrong vector"


def test_query_embeddings_max_time():
    vlc_db = ob.VlcDb(3)
    session_id = vlc_db.add_session(0)

    a_id = vlc_db.add_image(session_id, 0, None)
    vlc_db.update_embedding(a_id, np.array([0, 0, 0.9]))

    b_id = vlc_db.add_image(session_id, 10, None)
    vlc_db.update_embedding(b_id, np.array([0, 1, 0]))

    c_id = vlc_db.add_image(session_id, 20, None)
    vlc_db.update_embedding(c_id, np.array([0, 0, 1]))

    imgs, sims = vlc_db.query_embeddings_max_time(
        np.array([0, 0, 1]), 1, [session_id], 15
    )

    assert imgs[0].metadata.image_uuid == a_id

    imgs, sims = vlc_db.query_embeddings_max_time(
        np.array([0, 0, 1]), 2, [session_id], 15
    )

    assert imgs[0].metadata.image_uuid == a_id
    assert imgs[1].metadata.image_uuid == b_id

    imgs, sims = vlc_db.query_embeddings_max_time(
        np.array([0, 0, 1]), 3, [session_id], 15
    )

    assert len(imgs) == 2


def test_batch_query_embeddings():
    vlc_db = ob.VlcDb(3)
    session_id = vlc_db.add_session(0)

    a_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(a_id, np.array([0, 0, 1]))

    b_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(b_id, np.array([0, 1, 0]))

    c_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(c_id, np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)]))

    d_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(d_id, np.array([0, 0, 1]))

    imgs, sims = vlc_db.batch_query_embeddings(np.array([[0, 1, 0], [0, 0.2, 0.2]]), 1)
    assert len(imgs) == 2, "number of results different from number of query vectors"
    assert len(imgs[0]) == 1, "returned wrong number of matches for vector"
    assert imgs[0][0].metadata.image_uuid == b_id, (
        "Did not match correct image embedding"
    )
    assert imgs[1][0].metadata.image_uuid == c_id, (
        "Did not match correct image embedding"
    )


def test_batch_query_embeddings_uuid_filter():
    vlc_db = ob.VlcDb(3)
    session_id = vlc_db.add_session(0)

    a_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(a_id, np.array([0, 0, 1]))

    b_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(b_id, np.array([0, 1, 0]))

    c_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(c_id, np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)]))

    d_id = vlc_db.add_image(session_id, datetime.now(), None)
    vlc_db.update_embedding(d_id, np.array([0, 0, 0.9]))

    matches = vlc_db.batch_query_embeddings_uuid_filter(
        [a_id, b_id], 1, lambda q, v, s: True
    )

    assert len(matches) == 2
    assert len(matches[0]) == 1
    assert len(matches[1]) == 1

    assert matches[0][0][1].metadata.image_uuid == a_id
    assert matches[0][0][2].metadata.image_uuid == a_id

    assert matches[1][0][1].metadata.image_uuid == b_id
    assert matches[1][0][2].metadata.image_uuid == b_id

    matches = vlc_db.batch_query_embeddings_uuid_filter(
        [a_id, b_id], 1, lambda q, v, s: q.metadata.epoch_ns != v.metadata.epoch_ns
    )

    assert matches[0][0][2].metadata.image_uuid == d_id
    assert matches[1][0][2].metadata.image_uuid == c_id


def test_batch_query_embeddings_filter():
    vlc_db = ob.VlcDb(3)
    session_id = vlc_db.add_session(0)

    a_id = vlc_db.add_image(session_id, 0, None)
    vlc_db.update_embedding(a_id, np.array([0, 0, 1]))

    b_id = vlc_db.add_image(session_id, 10, None)
    vlc_db.update_embedding(b_id, np.array([0, 1, 0]))

    c_id = vlc_db.add_image(session_id, 20, None)
    vlc_db.update_embedding(c_id, np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)]))

    d_id = vlc_db.add_image(session_id, 30, None)
    vlc_db.update_embedding(d_id, np.array([0, 0, 0.9]))

    matches = vlc_db.batch_query_embeddings_filter(
        np.array([[0, 0, 1], [0, 1, 0]]), 1, lambda q, v, s: True
    )

    assert len(matches) == 2
    assert len(matches[0]) == 1
    assert len(matches[1]) == 1

    # assert matches[0][0][1].metadata.image_uuid == a_id
    assert matches[0][0][2].metadata.image_uuid == a_id

    # assert matches[1][0][1].metadata.image_uuid == b_id
    assert matches[1][0][2].metadata.image_uuid == b_id

    metadata = [vlc_db.get_image(a_id), vlc_db.get_image(b_id)]
    matches = vlc_db.batch_query_embeddings_filter(
        np.array([[0, 0, 1], [0, 1, 0]]),
        1,
        lambda q, v, s: q.metadata.epoch_ns != v.metadata.epoch_ns,
        filter_metadata=metadata,
    )

    assert matches[0][0][2].metadata.image_uuid == d_id
    assert matches[1][0][2].metadata.image_uuid == c_id

    matches = vlc_db.batch_query_embeddings_filter(
        np.array([[0, 1, 0]]),
        1,
        lambda q, v, s: v.metadata.epoch_ns < 10,
    )

    assert matches[0][0][2].metadata.image_uuid == a_id


def test_query_embeddings_filter():
    vlc_db = ob.VlcDb(3)
    session_id = vlc_db.add_session(0)

    a_id = vlc_db.add_image(session_id, 0, None)
    vlc_db.update_embedding(a_id, np.array([0, 0, 1]))

    b_id = vlc_db.add_image(session_id, 10, None)
    vlc_db.update_embedding(b_id, np.array([0, 1, 0]))

    c_id = vlc_db.add_image(session_id, 20, None)
    vlc_db.update_embedding(c_id, np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)]))

    d_id = vlc_db.add_image(session_id, 30, None)
    vlc_db.update_embedding(d_id, np.array([0, 0, 0.9]))

    matches = vlc_db.query_embeddings_filter(np.array([0, 0, 1]), 1, lambda v, s: True)
    assert len(matches) == 1
    assert len(matches[0]) == 2

    assert matches[0][1].metadata.image_uuid == a_id

    matches = vlc_db.query_embeddings_filter(
        np.array([0, 0, 1]),
        1,
        lambda v, s: v.metadata.epoch_ns != 0,
    )
    assert matches[0][1].metadata.image_uuid == d_id

    matches = vlc_db.query_embeddings_filter(
        np.array([0, 0, 1]),
        2,
        lambda v, s: v.metadata.epoch_ns != 0,
    )
    assert matches[0][1].metadata.image_uuid == d_id
    assert matches[1][1].metadata.image_uuid == c_id


def test_update_keypoints():
    vlc_db = ob.VlcDb(3)
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
    vlc_db = ob.VlcDb(3)
    robot_id = 0
    sensor_id = 42
    session_id = vlc_db.add_session(robot_id, sensor_id, "session_name")

    session = vlc_db.get_session(session_id)

    assert session.robot_id == 0
    assert session.name == "session_name"
    assert session.sensor_id == 42


def test_insert_session():
    vlc_db = ob.VlcDb(3)
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
    vlc_db = ob.VlcDb(3)
    session_id = vlc_db.add_session(0)

    computed_ts = datetime.now()
    loop_closure = ob.SparkLoopClosure(
        from_image_uuid=0,
        to_image_uuid=1,
        f_T_t=np.eye(4),
        quality=1,
        is_metric=True,
    )

    lc_uuid = vlc_db.add_lc(loop_closure, session_id, creation_time=computed_ts)
    lc = vlc_db.get_lc(lc_uuid)
    assert lc.from_image_uuid == 0
    assert lc.to_image_uuid == 1
    assert lc.f_T_t.shape == (4, 4)
    assert lc.is_metric
    assert lc.metadata.lc_uuid == lc_uuid
    assert lc.metadata.session_uuid == session_id


def test_iterate_lcs():
    vlc_db = ob.VlcDb(3)
    session_id = vlc_db.add_session(0)

    computed_ts = datetime.now()
    loop_closure = ob.SparkLoopClosure(
        from_image_uuid=0, to_image_uuid=1, f_T_t=np.eye(4), quality=1, is_metric=True
    )
    lc_uuid_1 = vlc_db.add_lc(loop_closure, session_id, creation_time=computed_ts)
    computed_ts_2 = datetime.now()
    loop_closure = ob.SparkLoopClosure(
        from_image_uuid=3,
        to_image_uuid=4,
        f_T_t=np.eye(4),
        quality=1,
        is_metric=False,
    )
    lc_uuid_2 = vlc_db.add_lc(loop_closure, session_id, creation_time=computed_ts_2)

    assert set((lc_uuid_1, lc_uuid_2)) == set(
        [lc.metadata.lc_uuid for lc in vlc_db.iterate_lcs()]
    )


def test_vlc_camera():
    """Test that camera matrix is correct."""
    camera = ob.PinholeCamera(5.0, 10.0, 3.0, 4.0)
    expected = np.array([[5.0, 0.0, 3.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]])
    assert camera.K == pytest.approx(expected)


def test_get_feature_depths():
    """Test that depth extraction for keypoints works."""
    img = ob.VlcImage(None, ob.SparkImage())
    # no keypoints -> no depths
    assert img.get_feature_depths() is None

    img.keypoints = np.array([[2, 1], [3.1, 1.9], [-1, 10], [10, -1]])
    # no depth image -> no depths
    assert img.get_feature_depths() is None

    img.image.depth = np.arange(24).reshape((4, 6))
    depths = img.get_feature_depths()
    # should be indices (1, 2), (2, 3), (3, 0), (0, 5)
    assert depths == pytest.approx(np.array([8, 15, 18, 5]))
