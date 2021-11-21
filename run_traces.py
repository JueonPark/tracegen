import os
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, help="traces/")
args = parser.parse_args()

print(args.path)

files = os.listdir(args.path)
directories = files - [f for f in files if isfile(join(args.path, f))]

"accel-sim.out -trace ./traces/ -config $SIM_CONFIG/%s/V100_downscaled_HBM2_PCI6x1/gpgpusim.config -config $SIM_CONFIG/%s/V100_downscaled_HBM2_PCI6x1/trace.config -cxl_config $SIM_CONFIG/%s/V100_downscaled_HBM2_PCI6x1/cxl.config -num_gpus 1 -num_cxl_memory_buffers 1"