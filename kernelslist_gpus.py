"""
copy trace to GPU_0
done after postprocessing
"""
import os
import argparse
from shutil import move

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help="traces/, make sure that '/' is added at the end")
parser.add_argument('--gpus', required=True, help="numboer of gpus")
args = parser.parse_args()

print(args.path)
print(args.gpus)

files = os.listdir(args.path)
for file in files:
    print(file)
    write_txt = file + ".traceg"
    for i in range(int(args.gpus) - 1):
        write_txt = write_txt + "\n" + file + ".traceg"
    write_txt = write_txt + "\ncudaDeviceSynchronize 0"
    print("content:")
    print(write_txt)
    print("path:")
    print(args.path + file + "/kernelslist.g", "w+")
    kernelslist = open(args.path + file + "/kernelslist.g", "w+")

    kernelslist.write(write_txt)
    kernelslist.close()