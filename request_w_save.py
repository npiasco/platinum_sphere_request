#!/home/npiasco/anaconda3/envs/py35/bin/python
import torch
import argparse
import PIL.Image
import torchvision.transforms as tf_torch
import dl_management.datasets.multmodtf as tf
from dl_management.networks.CustomArchi import DeploymentNet
import numpy as np
import logging
import dl_management.datasets.Platinum as Data
import os


logger = logging.getLogger()
logger.setLevel('ERROR')

# Activate virtual env -> https://stackoverflow.com/questions/6943208/activate-a-virtualenv-with-a-python-script


parser = argparse.ArgumentParser(description="Find the nearest RGBDL sphere")
parser.add_argument("input", metavar="Input_Image", help="Input Image File")
parser.add_argument("--db", default="data/default.db", help="Database file contening sphere signatures")
parser.add_argument("--net", default="data/default_net.pth", help="Net image descriptor to use")
parser.add_argument("--out_path", default="", help="Output location of the ranking results")
parser.add_argument("--out_file", default="scores", help="Output file name (final name will be given_name.csv)")
parser.add_argument("--old_version", dest='old_v', action='store_true', help="Using old network")
parser.add_argument("--dataset_root", default="", help="Dataset path")
parser.add_argument("--dataset_name", default="", help="Dataset file")
parser.add_argument("--dataset_split", default=False, help="Dataset split option", type=bool)
parser.set_defaults(old_v=False)


args = parser.parse_args()

#  Loading serialized data
db = torch.load(args.db)

transform = tf_torch.Compose(
    (
        tf_torch.Resize((224,224)),
        tf_torch.ToTensor(),
        tf_torch.Normalize(mean=[0.6406, 0.6457, 0.6116], std=[0.3180, 0.3017, 0.3233])
    )
)

if args.old_v:
    net = torch.load(args.net).cpu().eval()
else:
    weights = torch.load(args.net)
    net = DeploymentNet()
    #net.load_state_dict(weights)

query_signature = net(
    torch.autograd.Variable(
        transform(
            PIL.Image.open(
                args.input
            )
        ).unsqueeze(0),
        requires_grad=False
    )
).squeeze().data.numpy()


diff = [np.dot(query_signature, d_feat[0]) for d_feat in db]
sorted_index = list(np.argsort(diff))
output = [(db[i][1], diff[i]) for i in reversed(sorted_index)]
output_file = args.out_path + args.out_file + '.csv'
with open(output_file, 'w') as f:
    for l in output:
        f.write(str(l[0]) + ',' + str(l[1]) + '\n')


modtouse = ['rgb']
transform = {
    'first': (tf.Resize((224, 224)),),
    'rgb': (tf.RandomVerticalFlip(p=1.0),
            #tf.ToTensor(),
            #tf.Normalize(mean=[0.6406, 0.6457, 0.6116], std=[0.3180, 0.3017, 0.3233])
            ),
}

params  =  {
    'root': args.dataset_root,
    'file': args.dataset_name,
    'modalities': modtouse,
    'transform':transform
}

if not args.dataset_split:
    params['panorama_split'] = None


dataset = Data.Platinum(**params)
try:
    os.mkdir('print_res')
except FileExistsError:
    pass
saving_dir = 'print_res/' + args.input.replace('.jpg', '').split('/')[-1]
try:
    os.mkdir(saving_dir)
except FileExistsError:
    pass

request = PIL.Image.open(args.input)
request.save(saving_dir + '/request.jpg')

for i in range(5):
    dataset[output[i][0][0]]['rgb'].save(saving_dir + '/{}.jpg'.format(i))