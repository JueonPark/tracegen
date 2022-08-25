import os
import shutil
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--packet-size', type=int, required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--buffer', type=int, required=True)
parser.add_argument('--sync', type=bool, default=False, required=False)
parser.add_argument('--simd', type=int, required=True)
args = parser.parse_args()

GPU_OFFSET = 0x1000000000000000
CUDA_SYNC = 'cudaDeviceSynchronize 0'

def kernel_num(kernel_string):
    s = kernel_string.find('kernel-')
    kernel_ = kernel_string[s:s+11]
    return kernel_


EXP_NAME = f'packet_{args.packet_size}_buffer_{args.buffer}_gpu_{args.gpu}_sync_{1 if args.sync else 0}_simd_{args.simd}'
BASE_PATH = f'/home/jueonpark/tracegen/traces/{args.model}/{EXP_NAME}'
TRACE_PATH = f'/home/jueonpark/tracegen/traces/{args.model}/traces/packet_32_buffer_1_gpu_1_sync_0_simd_8'
HLO_PATH = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/{EXP_NAME}'
kernelslist_tmp_file = open(f'/home/jueonpark/tracegen/traces/{args.model}/traces/packet_32_buffer_1_gpu_1_sync_0_simd_8/kernelslist.g.tmp', 'r')

kernel_list = kernelslist_tmp_file.read().split('\n\n')
if kernel_list[-1] == "":
    kernel_list = kernel_list[:-1]
for ndp_gpu_kernels in tqdm(kernel_list):
    # print(ndp_gpu_kernels)
    kernels = ndp_gpu_kernels.split('\n')
    kernel_name = kernel_num(ndp_gpu_kernels)
    for gpu in range(args.gpu):
        os.makedirs(f'{BASE_PATH}/{kernel_name}/GPU_{gpu}')
    for kernel in kernels:
        if 'kernel-' in kernel:
            gpu_kernel = kernel
    kernelslist_file_base = open(f'{BASE_PATH}/{kernel_name}/kernelslist.g', 'w')
    for gpu in range(args.gpu):
        kernelslist_file_base.write(gpu_kernel + '\n')
    kernelslist_file_base.write(CUDA_SYNC)
    kernelslist_file_base.close()
    for kernel in kernels:
        # NDP kernel
        if 'ON_THE_FLY' in kernel or 'NDP' in kernel:
          ndp_kernel_path = f'{HLO_PATH}/{kernel}'
          # print(ndp_kernel_path)
          # case for NdpEwiseFusedOnTheFly
          if os.path.exists(ndp_kernel_path):
            shutil.copy(ndp_kernel_path, f'{BASE_PATH}/{kernel_name}/GPU_0')
        # GPU kernel
        if 'kernel-' in kernel:
            gpu_kernel_path = f'{TRACE_PATH}/{kernel}'
            shutil.copy(gpu_kernel_path, f'{BASE_PATH}/{kernel_name}/GPU_0')

    if args.sync and 'bn' in ndp_gpu_kernels:
      kernelslist_file_0 = open(f'{BASE_PATH}/{kernel_name}/GPU_0/kernelslist.g', 'w')
      kernelslist_file_1 = open(f'{BASE_PATH}/{kernel_name}/GPU_1/kernelslist.g', 'w')
      for kernel in kernels:
          if os.path.exists(f'{HLO_PATH}/{kernel}') or \
              kernel.startswith('kernel-') or \
              '_BAR_' in kernel:
              kernelslist_file_0.write(kernel + '\n')
              if 'pool' in kernel:
                  print('Max pool in '+kernel_name)
                  kernelslist_file_1.write(kernel+'\n')
                  os.symlink(f'../GPU_0/{kernel}', f'{BASE_PATH}/{kernel_name}/GPU_1/{kernel}')
      kernelslist_file_1.write(gpu_kernel)
      kernelslist_file_0.close()
      kernelslist_file_1.close()
      # symlink
      kernels_1 = os.listdir(f'{BASE_PATH}/{kernel_name}/GPU_1')
      for gpu in range(1, args.gpu):
        gpu_kernel_1 = os.symlink(f'../GPU_0/{gpu_kernel}', f'{BASE_PATH}/{kernel_name}/GPU_{gpu}/{gpu_kernel}')

      for gpu in range(2, args.gpu):
        kernels_gpu_path = f'{BASE_PATH}/{kernel_name}/GPU_{gpu}'
        for kernel_1 in kernels_1:
          os.symlink(f'../GPU_1/{kernel_1}', f'{kernels_gpu_path}/{kernel_1}')
    else:
      kernelslist_file = open(f'{BASE_PATH}/{kernel_name}/GPU_0/kernelslist.g', 'w')
      for kernel in kernels:
        if os.path.exists(f'{HLO_PATH}/{kernel}') or \
          kernel.startswith('kernel-') or \
          '_BAR_' in kernel:
          kernelslist_file.write(kernel + '\n')
      kernelslist_file.close()
      # symlink
      kernels_0 = os.listdir(f'{BASE_PATH}/{kernel_name}/GPU_0')
      for gpu in range(0, args.gpu):
          kernels_gpu_path = f'{BASE_PATH}/{kernel_name}/GPU_{gpu}'
          for kernel_0 in kernels_0:
            if gpu == 0:
              continue
            os.symlink(f'../GPU_0/{kernel_0}', f'{kernels_gpu_path}/{kernel_0}')
