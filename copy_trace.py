"""
copy trace to GPU_0
done after postprocessing
"""
import os
import argparse
from shutil import move

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help="traces/, make sure that '/' is added at the end")
args = parser.parse_args()

print(args.path)

files = os.listdir(args.path)
for file in files:
    # make directory for each trace
    if (file.find(".traceg") == -1):
        continue
    print("making directory for %s"%args.path+file[:-7])
    os.system('mkdir %s'%args.path + file[:-7])
    os.system('mkdir %s'%args.path + file[:-7]+"/GPU_0")
    # copy traceg file to 
    print("moving .traceg file to %s"%args.path + file[:-7])
    move(args.path + file, args.path + file[:-7]+"/GPU_0/")
    # write kernelslist.g to that directory
    write_txt = file + "\ncudaDeviceSynchronize 0"
    kernelslist = open(args.path + file[:-7]+"/kernelslist.g", "w+")
    kernelslist.write(write_txt)
    kernelslist.close()
    # write kernelslist.g to that directory's GPU_0
    kernelslist_gpu0 = open(args.path + file[:-7]+"/GPU_0/kernelslist.g", "w+")
    kernelslist_gpu0.write(file)
    kernelslist_gpu0.close()