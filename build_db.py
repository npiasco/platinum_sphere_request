#!/usr/bin/env python3
import torch
import argparse
import os
import SphereDataset as Data


# Activate virtual env -> https://stackoverflow.com/questions/6943208/activate-a-virtualenv-with-a-python-script


parser = argparse.ArgumentParser(description="Precompute signature of database sphere")
parser.add_argument("input", metavar="Input_Graph", help="Input Graph File")
parser.add_argument("--net", default="data/default_net.pth", help="Net image descriptor to use")

args = parser.parse_args()

#  Loading serialized data
net = torch.load(args.net)

modtouse = ['rgb']
transform = {
    'first': (Data.Resize((224,224)),),
    'rgb': (Data.ToTensor(), ),
}

root_to_folders = os.environ.get('PLATINUM', '/home/nathan/Dev/Code/platinum/') + 'data/'
dataset = Data.SphereDataset(root=root_to_folders,
                             file=args.input,
                             modalities=modtouse,
                             transform=transform)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=8)

dataset_feats = [(net(torch.autograd.Variable(example['rgb'],
                                                  requires_grad=False)).squeeze().data.numpy(),
                  example['idx'].numpy()) for example in dataloader]


torch.save(dataset_feats, 'data/default.db')