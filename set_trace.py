"""
after generating trace, set trace dir for eack kernel
"""
import os
import argparse
from shutil import copyfile
from shutil import copy

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help="trace directory")
args = parser.parse_args()

print(args.path)

files = os.listdir(args.path)
for file in files:
    # make directory for each trace
    if (file.find(".trace") == -1):
        continue
    print("making directory for %s"%args.path+file[:-6])
    os.system('mkdir %s'%args.path+file[:-6])
    os.system('mkdir %s'%args.path+file[:-6]+"/GPU_0")
# post processing for traces
os.system('$POST_PROCESSING %skernelslist'%args.path)

# rewrite kernelslist.g
copyfile(args.path+"kernelslist.g", args.path+"kernelslist.g.backup")

for file in files:
    if (file.find(".trace") == -1):
        continue
    print("moving .traceg file to %s"%args.path+file[:-6])
    copy(args.path+file+"g", args.path+file[:-6]+"/")
    copy(args.path+"kernelslist.g", args.path+file[:-6]+"/")
