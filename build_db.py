#!/home/npiasco/anaconda3/envs/py35/bin/python
import torch.utils.data
import argparse
import os
import dl_management.datasets.Platinum as Data
import dl_management.datasets.multmodtf as tf


# Activate virtual env -> https://stackoverflow.com/questions/6943208/activate-a-virtualenv-with-a-python-script


parser = argparse.ArgumentParser(description="Precompute signature of database sphere")
parser.add_argument("input", metavar="Input_Graph", help="Input Graph File")
parser.add_argument("--net", default="data/default_net.pth", help="Net image descriptor to use")

args = parser.parse_args()

#  Loading serialized data
net = torch.load(args.net).cpu().eval()

modtouse = ['rgb']
transform = {
    'first': (tf.Resize((224, 224)),),
    'rgb': (tf.ToTensor(), ),
}

root_to_folders = os.environ.get('PLATINUM', '/home/nathan/Dev/Code/platinum/')
dataset = Data.Platinum(root=root_to_folders,
                                 file=args.input,
                                 modalities=modtouse,
                                 transform=transform)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=8)

dataset_feats = [(net(torch.autograd.Variable(example['rgb'],
                                              requires_grad=False)).squeeze().data.numpy(),
                  example['idx'].squeeze().numpy()) for example in dataloader]

torch.save(dataset_feats, 'data/default.db')
