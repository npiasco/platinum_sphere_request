#!/home/npiasco/anaconda3/envs/py35/bin/python
import logging 
import torch.utils.data
import argparse
import dl_management.datasets.Platinum as Data
import dl_management.datasets.multmodtf as tf
from dl_management.networks.CustomArchi import DeploymentNet
import os 
import tqdm
logger = logging.getLogger()
logger.setLevel('ERROR')

parser = argparse.ArgumentParser(description="Precompute signature of database sphere")
parser.add_argument("input", metavar="Input_Graph", help="Input Graph File")
parser.add_argument("--net", default=None, help="Net image descriptor to use")
parser.add_argument("--root", default='', help="Data folder")
parser.add_argument("--jobs", default=8, help="Number of jobs")
parser.add_argument("--split", default=True, help="Split panoramic")
parser.add_argument("--out_path", default=None, help="Output location of the database signatures")
parser.add_argument("--out_file", default="default.db", help="Output file name")
parser.add_argument("--old_version", dest='old_v', action='store_true', help="Using old network")
parser.set_defaults(old_v=False)


args = parser.parse_args()
dir_path = os.path.dirname(os.path.realpath(__file__))

if args.net is None:
    args.net = dir_path + '/data/default_net.pth'
if args.out_path is None:
    args.out_path = dir_path + '/data/'


modtouse = ['rgb']
transform = {
    'first': (tf.Resize((224, 224)),),
    'rgb': (tf.ToTensor(),
            tf.Normalize(mean=[0.6406, 0.6457, 0.6116], std=[0.3180, 0.3017, 0.3233])
            ),
}

params  =  {
    'root':args.root,
    'file':args.input,
    'modalities': modtouse,
    'transform':transform
}

if args.split == 'False':
    params['panorama_split'] = None


dataset = Data.Platinum(**params)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=args.jobs)

if args.old_v:
    net = torch.load(args.net).cpu().eval()
else:
    weights = torch.load(args.net)
    net = DeploymentNet()
    net.load_state_dict(weights)

dataset_feats = [(net(torch.autograd.Variable(example['rgb'],
                                              requires_grad=False)).squeeze().data.numpy(),
                  example['idx'].squeeze().numpy()) for example in tqdm.tqdm(dataloader)]

torch.save(dataset_feats, args.out_path + args.out_file)
