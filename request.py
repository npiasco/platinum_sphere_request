#!/home/npiasco/anaconda3/envs/py35/bin/python
import torch
import argparse
import PIL.Image
import torchvision.transforms as tf
import numpy as np


# Activate virtual env -> https://stackoverflow.com/questions/6943208/activate-a-virtualenv-with-a-python-script


parser = argparse.ArgumentParser(description="Find the nearest RGBDL sphere")
parser.add_argument("input", metavar="Input_Image", help="Input Image File")
parser.add_argument("--db", default="data/default.db", help="Database file contening sphere signatures")
parser.add_argument("--net", default="data/default_net.pth", help="Net image descriptor to use")
parser.add_argument("--out_path", default=".", help="Output location of the ranking results")
parser.add_argument("--out_file", default="scores", help="Output file name (+.csv)")

args = parser.parse_args()

#  Loading serialized data
net = torch.load(args.net).cpu().eval()
db = torch.load(args.db)

transform = tf.Compose(
    (
        tf.Resize((224,224)),
        tf.ToTensor()
    )
)

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
