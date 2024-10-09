from models import basicnet
def load_model(model_name):
    if model_name == 'basicnet':
        return basicnet.BasicNet()