import os
import sys

sys.path.append('..')
from models.model import HASNet

# train path
DUTS_TR = './datasets/DUTS/Train/'

# test path
DUTOMRON = './datasets/DUT-OMRON/Test/'
ECSSD = './datasets/ECSSD/Test/'
HKU_IS = './datasets/HKU-IS/Test/'
PASCALS = './datasets/PASCAL-S/Test/'
DUTS_TE = './datasets/DUTS/Test/'

test_datasets = {'DUTS-TE': DUTS_TE, 'ECSSD': ECSSD, 'HKU-IS': HKU_IS, 'PASCAL-S': PASCALS, 'DUT-OMRON': DUTOMRON, }
results_path = './results/'
results_save_path = os.path.join('.', 'results/')

image_root = './datasets/DUTS/Train/'
# image_list = './data/DUTS/DUTS-TR/train_pair.lst'
train_size = 256
batch_size = 16
num_workers = 12
ckpt_path = './saved_models/MobileBaseline/'
config = {
    'image_root': image_root,
    # 'image_list': image_list,
    'train_size': train_size,
    'test_size': train_size,
    'batch_size': batch_size,
    'num_workers': num_workers,
    'mode': 'train',
    'shuffle': True,
    'epochs': 100,
    'lr': 1e-4,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'pin_memory': True,
    'decay_rate': 0.1,
    'decay_epoch': 50,
    'means': [0.485, 0.456, 0.406],
    'stds': [0.229, 0.224, 0.225],
}


def getNet():
    return HASNet().cuda()


def infer(net, inputs):
    side_out4, side_out3, side_out2, side_out1 = net(inputs)
    return side_out1

if __name__ == '__main__':
    print(config['image_root'])
