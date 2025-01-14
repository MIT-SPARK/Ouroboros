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

    def query(self, embeddings, k=1, similarity_metric="ip"):
        if similarity_metric == "ip":
            similarities = embeddings @ self._vectors.T

        elif similarity_metric == "cos":
            raise NotImplementedError(
                "Cosine similarity not yet implemented as similarity metric"
            )
        elif callable(similarity_metric):
            similarities = np.zeros((len(embeddings), len(self._vectors)))
            for ex, e in enumerate(embeddings):
                for vx, v in enumerate(self._vectors):
                    similarities[ex, vx] = similarity_metric(e, v)

        else:
            raise NotImplementedError(
                f"InvertibleVectorStore does not implement similarity metric {similarity_metric}"
            )

        uuid_list = []
        similarity_list = []
        for row in range(similarities.shape[0]):
            if k > 0:
                indices = np.squeeze(-similarities[row, :]).argsort()[:k]
            else:
                indices = np.squeeze(-similarities[row, :]).argsort()
            uuid_list.append([self._local_idx_to_uuid[i] for i in indices])

            if k > 0:
                max_similarities = -np.sort(-similarities[row, :])[:k]
            else:
                max_similarities = -np.sort(-similarities[row, :])
            similarity_list.append(max_similarities)

        return uuid_list, similarity_list
