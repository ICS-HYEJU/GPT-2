import os
import torch

def load_network(net, device, weight_path):
    assert os.path.exits(weight_path), f'There is no saved weight file...'
    print('loading the model form %s weigth_path')
    state_dict = torch.load(weight_path, map_location=str(device))
    net.load_state_dict(state_dict['model'])
    print('load completed...')
    return net