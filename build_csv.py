#!/home/npiasco/anaconda3/envs/py35/bin/python
import argparse
import os
import pathlib as path
import tqdm
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description="Precompute csv file")
parser.add_argument("input", metavar="Data_folder", help="Path to the data folder containing the data")
parser.add_argument("--output", default="dataset", help="Name of the output file")
parser.add_argument("--prune", default=2, help="Pruning factor", type=int)
parser.add_argument("--xml", dest='xml', action='store_true', help="Xml info (for stereopolis perspective images)")
parser.set_defaults(xml=False)

args = parser.parse_args()
dir_path = os.path.dirname(os.path.realpath(__file__))

if args.xml:
    im_name = 'png/'
else:
    im_name = 'rgb/'

p = path.Path(args.input + im_name)
rgb_files = [im_name + file.name for i, file in enumerate(sorted(p.iterdir())) if file.is_file() and
             '-300-' not in file.name and i%args.prune == 0]

if args.xml:
    depth_name = 'png/'
else:
    depth_name = 'depth/'

p = path.Path(args.input + depth_name)
depth_files = [depth_name + file.name for i, file in enumerate(sorted(p.iterdir())) if file.is_file() and
              i%args.prune == 0]

if args.xml:
    sem_name = 'png/'
else:
    sem_name = 'intensity/'

p = path.Path(args.input + sem_name)
sem_files = [sem_name + file.name for i, file in enumerate(sorted(p.iterdir())) if file.is_file() and
             i%args.prune == 0]

coords = list()
if args.xml:
    info_name = 'oriXml/'
else:
    info_name = 'info/'

p = path.Path(args.input + info_name)
for file in tqdm.tqdm(sorted(p.iterdir())):
    if file.is_file():
        if args.xml:
            tree = ET.parse(args.input + info_name + file.name)
            root = tree.getroot()
            easting = root[2][0][2][0].text
            northing = root[2][0][2][1].text
            im_name = root[1][0].text
            if 'png/' + im_name + '.png' in rgb_files:
                coords.append([float(easting), float(northing)])
        else:
            with open(args.input + info_name + file.name, 'r') as f:
                f.readline()
                coordfull = f.readline()
                clean_coord = coordfull.replace('[', '').replace(']', '').split(',')
                coord = [float(clean_coord[0]), float(clean_coord[1])]
                coords.append(coord)

with open(args.output + '.csv', 'w') as f:
    f.write('# File generated with npiasco code\n')
    f.write('# From folder {}\n'.format(args.input))
    for i, coord in enumerate(coords):
        f.write('{};{};{};{};{};{}\n'.format(i, coord[0], coord[1],
                                             rgb_files[i], depth_files[i], sem_files[i]))

print('Done.')



