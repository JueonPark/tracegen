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

# make directory for each trace
# os.system('mkdir %s'%args.path+"GPU_1")
# os.system('mkdir %s'%args.path+file+"/GPU_2")
# os.system('mkdir %s'%args.path+file+"/GPU_3")
# copy traceg file to 
print("moving GPU_0 to %s"%args.path+"GPUs")
os.system("cp -R %s %s"%(args.path+"GPU_0/*", args.path+"GPU_1"))
# os.system("cp -R %s %s"%(args.path+file+"/GPU_0/*", args.path+file+"/GPU_2"))
# os.system("cp -R %s %s"%(args.path+file+"/GPU_0/*", args.path+file+"/GPU_3"))