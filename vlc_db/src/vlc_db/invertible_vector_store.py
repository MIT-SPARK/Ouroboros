import numpy as np


class InvertibleVectorStore:
    # TODO: Replace with this? https://github.com/nmslib/hnswlib/
    def __init__(self, n_dim, metric="ip"):
        self._metric = metric
        self._vectors = np.empty((0, n_dim))
        self._uuid_to_vector = {}
        self._local_idx_to_uuid = {}

    def __getitem__(self, key):
        return self._uuid_to_vector[key]

    def __setitem__(self, key, newval):
        self._uuid_to_vector[key] = newval
        self._vectors = np.vstack([self._vectors, newval])
        idx = len(self._vectors) - 1
        self._local_idx_to_uuid[idx] = key

    def query(self, embeddings, k=1):
        if self._metric == "ip":
            dists = -self._vectors @ embeddings.T
            uuid_list = []
            distance_list = []
            for col in range(dists.shape[1]):
                indices = np.squeeze(dists[:, col]).argsort()[:k]
                uuid_list.append([self._local_idx_to_uuid[i] for i in indices])

                min_dists = np.sort(np.squeeze(dists[:, col]))[:k]
                distance_list.append(min_dists)

            return uuid_list, distance_list
        else:
            raise NotImplementedError(
                "InvertibleVectorStore only implements inner product and query method"
            )
