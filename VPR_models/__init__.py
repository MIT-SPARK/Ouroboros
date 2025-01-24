class VPR_model:
    
    def VPR_model(self):
        self.descriptor_dim = None
        self.ckpt_path = None
        self.model = None

    def __call__(self, input_data):
        return self.model(input_data)