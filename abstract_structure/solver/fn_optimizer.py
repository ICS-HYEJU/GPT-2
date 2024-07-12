import torch

def build_optimizer(cfg, model):
    if cfg['solver']['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg['solver']['lr0'],
                                     weight_decay=cfg['solver']['weight_decay'])
    else:
        raise NotImplementedError('{} not implemented'.format(cfg['solver']['name']))

    return optimizer