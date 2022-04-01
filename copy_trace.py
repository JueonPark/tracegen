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

# TODO: handle multi-gpu situation
files = os.listdir(args.path)
for file in files:
    # make directory for each trace
    if (file.find(".traceg") == -1):
        continue
    print("making directory for %s"%args.path+file[:-7])
    kernel_dir = os.path.join(args.path, file[:-7])
    print("kernel dir:" + kernel_dir)
    os.makedirs(kernel_dir, 0o777)
    for i in range(int(args.gpus)):
        gpu_dir = kernel_dir + "/GPU_"+str(i)
        print("gpu dir:" + gpu_dir)
        os.makedirs(gpu_dir, 0o777)
    # copy traceg files to each kernel dir
    print("moving .traceg file to %s"%kernel_dir)
    for i in range(int(args.gpus)):
        gpu_dir = kernel_dir + "/GPU_"+str(i)
        move(args.path + file, gpu_dir)
    # write kernelslist.g to that directory
    write_txt = file
    for i in range(int(args.gpus) - 1):
        write_txt = write_txt + "\n" + file
    write_txt = write_txt + "\ncudaDeviceSynchronize 0"
    kernelslist = open(args.path + file[:-7]+"/kernelslist.g", "w+")
    kernelslist.write(write_txt)
    kernelslist.close()
    # write kernelslist.g to that directory's GPU_0
    kernelslist_gpu0 = open(args.path + file[:-7]+"/GPU_0/kernelslist.g", "w+")
    kernelslist_gpu0.write(file)
    kernelslist_gpu0.close()