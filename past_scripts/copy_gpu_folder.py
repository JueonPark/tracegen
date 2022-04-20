"""
copy trace to GPU_0
done after postprocessing
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help="traces/, make sure that '/' is added at the end")
args = parser.parse_args()

print(args.path)

files = os.listdir(args.path)
for file in files:
    # make directory for each trace
    os.system('mkdir %s'%args.path+file+"/GPU_1")
    os.system('mkdir %s'%args.path+file+"/GPU_2")
    os.system('mkdir %s'%args.path+file+"/GPU_3")
    # copy traceg file to 
    print("moving GPU_0 to %s"%args.path+file+"/GPUs")
    os.system("cp -R %s %s"%(args.path+file+"/GPU_0/*", args.path+file+"/GPU_1"))
    os.system("cp -R %s %s"%(args.path+file+"/GPU_0/*", args.path+file+"/GPU_2"))
    os.system("cp -R %s %s"%(args.path+file+"/GPU_0/*", args.path+file+"/GPU_3"))
    # write kernelslist.g to that directory
    write_txt = file + "\ncudaDeviceSynchronize 0"
    kernelslist = open(args.path + file+"/kernelslist.g", "w+")
    kernelslist.write(write_txt)
    kernelslist.close()
    # write kernelslist.g to that directory's GPU_0
    kernelslist_gpu0 = open(args.path + file+"/GPU_0/kernelslist.g", "w+")
    kernelslist_gpu0.write(file)
    kernelslist_gpu0.close()
