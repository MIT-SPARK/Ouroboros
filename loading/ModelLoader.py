def load_model(config):
    if config.VPR_model == 'CliqueMining':
        from VPR_models.CliqueMining import CliqueMining
        return CliqueMining(config.ckpt_path, config.device)
    else:
        raise ValueError(f"Model {config.model} not supported")