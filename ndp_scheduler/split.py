import os
import argparse
import shutil
from utils import parse_stats
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--pivot', type=int, required=True)
args = parser.parse_args()

kernelslist = open(os.path.join(args.model, 'traces', 'kernelslist.g'), 'r').read().split('\n')
if kernelslist[-1] == "":
    kernelslist = kernelslist[:-1]

forward_path = os.path.join(args.model, 'traces_fw')
backward_path = os.path.join(args.model, 'traces_bw')
os.mkdir(forward_path)
os.mkdir(backward_path)

dst_path = forward_path
kernelslist_fw_file = open(os.path.join(forward_path, 'kernelslist.g'), 'w')
kernelslist_bw_file = open(os.path.join(backward_path, 'kernelslist.g'), 'w')
write_file = kernelslist_fw_file
for kernel_name in tqdm(kernelslist):
    src_path = os.path.join(args.model, 'traces', kernel_name)
    write_file.write(kernel_name + '\n')
    if str(args.pivot) in kernel_name:
        dst_path = backward_path
        write_file = kernelslist_bw_file
    shutil.copy(src_path, dst_path)
