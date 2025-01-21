from abc import ABC, abstractmethod

class MatcherBase(ABC):
    @abstractmethod
    def __init__(self, max_num_keypoints=2048, device=None):
        pass

    @abstractmethod
    def load_image(self, img_path, resize):
        pass

    @abstractmethod
    def execute(self, img0, img1):
        pass

    @abstractmethod
    def plot(self, img0, img1, m_kpts0, m_kpts1, kpts0, kpts1):
        pass
