import numpy as np


class InvertibleVectorStore:
    # TODO: Replace with this? https://github.com/nmslib/hnswlib/
    def __init__(
        self,
        n_dim,
    ):
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

    def query(self, embeddings, k=1, distance_metric="ip"):
        if distance_metric == "ip":
            distances = embeddings @ self._vectors.T

        elif distance_metric == "cos":
            raise NotImplementedError(
                "Cosine similarity not yet implemented as distance metric"
            )
        elif callable(distance_metric):
            distances = np.zeros((len(embeddings), len(self._vectors)))
            for ex, e in enumerate(embeddings):
                for vx, v in enumerate(self._vectors):
                    distances[ex, vx] = distance_metric(e, v)

        else:
            raise NotImplementedError(
                f"InvertibleVectorStore does not implement distance metric {distance_metric}"
            )

        uuid_list = []
        distance_list = []
        for row in range(distances.shape[0]):
            if k > 0:
                indices = np.squeeze(distances[row, :]).argsort()[:k]
            else:
                indices = np.squeeze(distances[row, :]).argsort()
            uuid_list.append([self._local_idx_to_uuid[i] for i in indices])

            if k > 0:
                min_dists = np.sort(distances[row, :])[:k]
            else:
                min_dists = np.sort(distances[row, :])
            distance_list.append(min_dists)

        return uuid_list, distance_list
