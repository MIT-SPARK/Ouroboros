from cliquemining.vpr_model import VPRModel
from VPR_models import VPR_model
import logging
import torch

class CliqueMining(VPR_model):
    
    def __init__(self, ckpt_path, device):

        self.descriptor_dim = 8192+256
        self.model = VPRModel(
            backbone_arch='dinov2_vitb14',
            backbone_config={
                'num_trainable_blocks': 4,
                'return_token': True,
                'norm_layer': True,
            },
            agg_arch='SALAD',
            agg_config={
                'num_channels': 768,
                'num_clusters': 64,
                'cluster_dim': 128,
                'token_dim': 256,
            },
        )

        self.model.load_state_dict(torch.load(ckpt_path,map_location=torch.device(device))['state_dict'])
        self.model = self.model.eval()
        self.model.to(device)
        logging.info("Model loaded successfully")