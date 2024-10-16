import torch
import mnist.models.convnet as convnet

def load(model_name, model_file=None):
    if model_name == 'ConvNet':
        net = convnet.ConvNet()
    else:
        print("i did not make that net and i humbly apologize")
    if model_file:
        net.load_state_dict(torch.load(model_file))
    return net