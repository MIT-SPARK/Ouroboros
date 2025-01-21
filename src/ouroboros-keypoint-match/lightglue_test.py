from keypoint_matching.common.keypoint_matcher import KeypointMatcher
from keypoint_matching.lightglue.lightglue_matcher import LightGlueMatcher

kpt_matcher = KeypointMatcher()
kpt_matcher.set_matcher(LightGlueMatcher(max_num_keypoints=1024))

img0 = kpt_matcher.load_image("./assets/image0.jpg")
img1 = kpt_matcher.load_image("./assets/image1.jpg")
kpts0, kpts1, m_kpts0, m_kpts1 = kpt_matcher.execute(img0, img1)

kpt_matcher.plot(img0, img1, m_kpts0, m_kpts1, kpts0, kpts1)

