import pytest
import uuid
from ouroboros.vlc_db.invertible_vector_store import InvertibleVectorStore
import numpy as np


def test_get_set_contains():
    store = InvertibleVectorStore(3)

    img1_uuid = uuid.uuid4()
    store.set(img1_uuid, np.array([0, 0, 1]))
    img2_uuid = uuid.uuid4()
    store.set(img2_uuid, np.array([1, 0, 0]))

    with pytest.raises(Exception):
        store.set(uuid.uuid4(), np.array([[0]]))

    assert np.all(store.get(img1_uuid) == np.array([0, 0, 1]))
    assert np.all(store.get(img2_uuid) == np.array([1, 0, 0]))

    with pytest.raises(Exception):
        store.get("key-doesn't-exist")

    assert store.contains(img1_uuid)
    assert not store.contains(uuid.uuid4())


def test_query():

    store = InvertibleVectorStore(3)

    a_id = uuid.uuid4()
    store.set(a_id, np.array([0, 0, 1]))

    b_id = uuid.uuid4()
    store.set(b_id, np.array([0, 1, 0]))

    c_id = uuid.uuid4()
    store.set(c_id, np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)]))

    d_id = uuid.uuid4()
    store.set(d_id, np.array([0, 0, 1]))

    imgs, sims = store.query(np.array([[0, 0.2, 0.8]]), 1)
    assert len(imgs) == 1, "query_embeddings returned wrong number of closest matches"
    assert len(imgs[0]) == 1
    assert len(sims) == 1
    assert len(sims[0]) == 1
    assert imgs[0][0] in [a_id, d_id]

    imgs, sims = store.query(np.array([[0, 0.2, 0.8]]), 2)
    assert len(imgs[0]) == 2, "query_embeddings returned wrong number of matches"
    assert set((a_id, d_id)) == set(imgs[0]), "Query did not match k closest"

    imgs, sims = store.query(np.array([[0, 0.2, 0.8]]), 3)
    assert len(imgs[0]) == 3, "query_embeddings returned wrong number of matches"
    assert set((a_id, c_id, d_id)) == set(imgs[0]), "Query did not match k closest"
