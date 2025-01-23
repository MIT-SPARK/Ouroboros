from .matcher_base import MatcherBase

class KeypointMatcher:
    def set_matcher(self, matcher: MatcherBase):
        self.matcher = matcher

    def load_image(self, img_path, resize=None):
        """
        Loads the images for image matching in the desired size.
            Arguments:
                img_path (str): Path to the image to be matched.
                resize (tuple of int, optional): Desired size to resize the images to, specified as (height, width).
        """
        return self.matcher.load_image(img_path, resize)
    
    def execute(self, img0, img1):
        """
        Performs image matching using the selected strategy.
            Arguments:
                img0 (Tensor): First image to be matched.
                img1 (Tensor): Second image to be matched.
                max_num_keypoints (int): Maximum number of keypoints to be found.
                device (str, optional): The device to run the matching on, either "cuda" or "cpu".
                
            Returns:
                kpts0 (Tensor): Keypoints detected in the first image after feature extraction.
                kpts1 (Tensor): Keypoints detected in the second image after feature extraction.
                m_kpts0 (Tensor): Matched keypoints from the first image.
                m_kpts1 (Tensor): Matched keypoints from the second image.
        """
        return self.matcher.execute(img0, img1)

    def plot(self, img0, img1, m_kpts0, m_kpts1, kpts0, kpts1):
        """
        Plots the matched keypoints.
            Arguments:
                img0 (Tensor): First image to be plotted.
                img1 (Tensor): Second image to be plotted.
                m_kpts0 (Tensor): Matched keypoints from the first image.
                m_kpts1 (Tensor): Matched keypoints from the second image.
        """
        self.matcher.plot(img0, img1, m_kpts0, m_kpts1, kpts0, kpts1)

    def get_matcher_model(self):
        """
        Returns the model of the keypoint matcher
        """
        return self.matcher.get_matcher_model()

    def get_descriptor_model(self):
        """
        Returns the model of the interest point detector
        """
        return self.matcher.get_descriptor_model()
