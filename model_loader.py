import os
import cifar10.model_loader
import mnist.basicnet

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    if dataset == 'mnist':
        net = mnist.basicnet()
    return net
