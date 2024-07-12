import torch

def build_scheduler(cfg, optimizer):
    if cfg.scheduler['name'] == 'LambdaLR':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=cfg.scheduler['lr_lambda'])
    else:
        raise NotImplementedError('{} is not Implemeted'.format(cfg['scheduler']['name']))
    return scheduler