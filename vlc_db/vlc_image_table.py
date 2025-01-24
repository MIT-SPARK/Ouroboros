from datetime import datetime
from typing import Union
import uuid
import numpy as np

from vlc_db.spark_image import SparkImage
from vlc_db.vlc_image import VlcImage, VlcImageMetadata
from vlc_db.invertible_vector_store import InvertibleVectorStore
from vlc_db.utils import epoch_ns_from_datetime

import faiss
import torch
from collections import deque

class VlcImageTable:
    def __init__(self, config, descriptor_dim):
        self.metadata_store = {}
        self.image_store = {}
        self.faiss_idx_to_uuid = {}
        self.keypoints_store = {}
        self.descriptors_store = {}
        # Initialize FAISS index
        flat_index = faiss.IndexFlatIP(descriptor_dim)
        index = faiss.IndexIDMap(flat_index)
        if config.device == 'cpu':
            self.index = index
        else:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, index)
        self.cur_id=0

        self.descriptor_cache = deque()
        self.config = config

    def get_image(self, image_uuid):
        if image_uuid not in self.metadata_store:
            raise Exception(f"Image {image_uuid} not found in VLC DB!")
        metadata = self.metadata_store[image_uuid]
        data = np.load(self.config.save_path+image_uuid+".npz",allow_pickle=True)
        image = SparkImage(data['rgb'] if data['rgb'].ndim!=0 else None,data['depth'] if data['depth'].ndim!=0 else None)
        embedding = None # TODO: fix
        # embedding = self.index.reconstruct(metadata.faiss_id)
        keypoints = None
        descriptors = None

        vlc_image = VlcImage(metadata, image, embedding, keypoints, descriptors)
        return vlc_image

    def get_image_keys(self):
        ts_keys = [
            (metadata.epoch_ns, key) for key, metadata in self.metadata_store.items()
        ]
        return [key for _, key in sorted(ts_keys)]

    def iterate_images(self):
        """Iterate through images according to ascending timestamp"""
        for key in self.get_image_keys():
            yield self.get_image(key)

    def query_embeddings_uuids(
        self,
        embeddings: np.ndarray,
        k: int
    ) -> ([[str]], [[float]]):
        """Embeddings is a NxD numpy array, where N is the number of queries and D is the descriptor size
        Queries for the top k matches.

        Returns the top k closest matches and the match distances
        """

        IPs, idxs = self.index.search(embeddings,k)
        return idxs[0], IPs[0]

    def query_embeddings(
        self,
        embeddings: np.ndarray,
        k: int
    ) -> ([[VlcImage]], [[float]]):
        """Embeddings is a NxD numpy array, where N is the number of queries and D is the descriptor size
        Queries for the top k matches.

        Returns the top k closest matches and the match distances (or less if < k images have been added)
        """

        idxs, IPs = self.query_embeddings_uuids(embeddings, k)
        images = []
        for idx in idxs:
            if idx!=-1:
                images.append(self.get_image(self.faiss_idx_to_uuid[idx]))

        return images, IPs[:len(images)]

    def add_image(
        self,
        session_id: str,
        image_timestamp: Union[int, datetime],
        image: SparkImage
    ) -> str:
        new_uuid = str(uuid.uuid4())

        if isinstance(image_timestamp, datetime):
            image_timestamp = epoch_ns_from_datetime(image_timestamp)

        metadata = VlcImageMetadata(
            image_uuid=new_uuid,
            session_id=session_id,
            epoch_ns=image_timestamp,
            faiss_id=self.cur_id
        )
        self.metadata_store[new_uuid] = metadata
        self.faiss_idx_to_uuid[self.cur_id] = new_uuid
        self.cur_id+=1
        
        # Save image to desk
        np.savez(self.config.save_path+new_uuid+".npz",rgb=image.rgb,depth=image.depth)

        return new_uuid

    def update_embedding(self, image_uuid: str, embedding):
        id = self.metadata_store[image_uuid].faiss_id
        self.index.add_with_ids(embedding,np.array([id]))
            

    def update_keypoints(self, image_uuid: str, keypoints, descriptors=None):
        self.keypoints_store[image_uuid] = keypoints
        if descriptors is not None:
            assert len(keypoints) == len(descriptors)
        self.descriptors_store[image_uuid] = descriptors

    def get_keypoints(self, image_uuid: str):
        return self.keypoints_store[image_uuid], self.descriptors_store[image_uuid]

    def drop_image(self, image_uuid: str):
        """This functionality is for marginalization / sparsification of history"""
        # TODO: delete things other than the FAISS occurence to save memory
        self.index.remove_ids(np.array([self.metadata_store[image_uuid].faiss_id]))
