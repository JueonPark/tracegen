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
parser.add_argument('--passes', type=str, required=True)
args = parser.parse_args()

GPU_OFFSET = 0x1000000000000000
CUDA_SYNC = 'cudaDeviceSynchronize 0'

def kernel_num(kernel_string):
    s = kernel_string.find('kernel-')
    kernel_ = kernel_string[s:s+11]
    return kernel_

passes = []
if args.passes == "fw":
    passes.append("fw")
elif args.passes == "bw":
    passes.append("bw")
elif args.passes == "all":
    passes.append("fw")
    passes.append("bw")
else:
    raise Exception("`sync` should be among `fw`, `bw` or `all`")

EXP_NAME = f'packet_{args.packet_size}_buffer_{args.buffer}_gpu_{args.gpu}_sync_{1 if args.sync else 0}_simd_{args.simd}'

BASE_PATHS = [f'{args.model}/{EXP_NAME}_{p}' for p in passes]
TRACES_PATHS = [f'{args.model}/traces_{p}/{EXP_NAME}_{p}' for p in passes]
HLO_PATHS = [f'{args.model}/xla_hlo_{p}/{EXP_NAME}' for p in passes]
kernelslist_tmp_files = [open(f'{args.model}/traces_{p}/{EXP_NAME}_{p}/kernelslist.g.tmp.{p}', 'r') for p in passes]

for p, kernelslist_tmp_file, base_path, traces_path, hlo_path in zip(passes, kernelslist_tmp_files, BASE_PATHS, TRACES_PATHS, HLO_PATHS):
    kernel_list = kernelslist_tmp_file.read().split('\n\n')
    if kernel_list[-1] == "":
        kernel_list = kernel_list[:-1]
    for ndp_gpu_kernels in tqdm(kernel_list):
        kernels = ndp_gpu_kernels.split('\n')
        kernel_name = kernel_num(ndp_gpu_kernels)
        for gpu in range(args.gpu):
            os.makedirs(f'{base_path}/{kernel_name}/GPU_{gpu}')
        for kernel in kernels:
            if 'kernel-' in kernel:
                gpu_kernel = kernel
        kernelslist_file_base = open(f'{base_path}/{kernel_name}/kernelslist.g', 'w')
        for gpu in range(args.gpu):
            kernelslist_file_base.write(gpu_kernel + '\n')
        kernelslist_file_base.write(CUDA_SYNC)
        kernelslist_file_base.close()
        for kernel in kernels:
            # NDP kernel
            if 'ON_THE_FLY' in kernel or 'NDP' in kernel:
                ndp_kernel_path = f'{hlo_path}/{kernel}'
                if 'fw' == p:
                    if not os.path.exists(ndp_kernel_path):
                        print('Warning: ', ndp_kernel_path)
                        on_the_fly_kernel_path = ndp_kernel_path.replace('NDP', 'ON_THE_FLY')
                        on_the_fly_file = open(on_the_fly_kernel_path, 'r')
                        not_on_the_fly_file = open(ndp_kernel_path, 'w')
                        not_on_the_fly_file.write(on_the_fly_file.read().replace(f"STG.{args.packet_size} v0 BASE OFFSET\n", "####\n"))
                        on_the_fly_file.close()
                        not_on_the_fly_file.close()
                    shutil.copy(ndp_kernel_path, f'{base_path}/{kernel_name}/GPU_0')
                elif 'bw' == p:
                    if not os.path.exists(ndp_kernel_path):
                        print('Warning: ', ndp_kernel_path)
                    else:
                        shutil.copy(ndp_kernel_path, f'{base_path}/{kernel_name}/GPU_0')
            # GPU kernel
            if 'kernel-' in kernel:
                gpu_kernel_path = f'{traces_path}/{kernel}'
                #shutil.copy(gpu_kernel_path, f'{base_path}/{kernel_name}/GPU_0')
                os.link(gpu_kernel_path, f'{base_path}/{kernel_name}/GPU_0/{kernel}')

        if args.sync and 'bn' in ndp_gpu_kernels:
            kernelslist_file_0 = open(f'{base_path}/{kernel_name}/GPU_0/kernelslist.g', 'w')
            kernelslist_file_1 = open(f'{base_path}/{kernel_name}/GPU_1/kernelslist.g', 'w')
            for kernel in kernels:
                if os.path.exists(f'{hlo_path}/{kernel}') or \
                   kernel.startswith('kernel-') or \
                   '_BAR_' in kernel:
                    kernelslist_file_0.write(kernel + '\n')
                    if 'pool' in kernel:
                        print('Max pool in '+kernel_name)
                        kernelslist_file_1.write(kernel+'\n')
                        os.symlink(f'../GPU_0/{kernel}', f'{base_path}/{kernel_name}/GPU_1/{kernel}')
            kernelslist_file_1.write(gpu_kernel)
            kernelslist_file_0.close()
            kernelslist_file_1.close()
            # symlink
            kernels_1 = os.listdir(f'{base_path}/{kernel_name}/GPU_1')
            for gpu in range(1, args.gpu):
                gpu_kernel_1 = os.symlink(f'../GPU_0/{gpu_kernel}', f'{base_path}/{kernel_name}/GPU_{gpu}/{gpu_kernel}')

            for gpu in range(2, args.gpu):
                kernels_gpu_path = f'{base_path}/{kernel_name}/GPU_{gpu}'
                for kernel_1 in kernels_1:
                    os.symlink(f'../GPU_1/{kernel_1}', f'{kernels_gpu_path}/{kernel_1}')
        else:
            kernelslist_file = open(f'{base_path}/{kernel_name}/GPU_0/kernelslist.g', 'w')
            for kernel in kernels:
                if os.path.exists(f'{hlo_path}/{kernel}') or \
                   kernel.startswith('kernel-') or \
                   '_BAR_' in kernel:
                    kernelslist_file.write(kernel + '\n')
            kernelslist_file.close()
            # symlink
            kernels_0 = os.listdir(f'{base_path}/{kernel_name}/GPU_0')
            for gpu in range(0, args.gpu):
                kernels_gpu_path = f'{base_path}/{kernel_name}/GPU_{gpu}'
                for kernel_0 in kernels_0:
                    if p == 'bw' and 'ph3' in kernel_0 and 'bn' in kernel_0:
                        # edit and copy
                        bn3_kernel = open(f'{base_path}/{kernel_name}/GPU_0/{kernel_0}', 'r')
                        bn3_content = bn3_kernel.read()
                        bn3_kernel_revised = open(f'{kernels_gpu_path}/{kernel_0}', 'w')
                        lines = bn3_content.split('\n')
                        for line_num, line in enumerate(lines):
                            if 'STGG' in line:
                                # parse line and add gpu offset
                                words = [word for word in lines[line_num-1].split(' ') if '0x' in word]
                                assert len(words) == 1
                                bn3_kernel_revised.write(f'STGG.32 v7 {hex(gpu * GPU_OFFSET + int(words[0], base=16))} OFFSET\n')
                            else:
                                bn3_kernel_revised.write(line + '\n')
                        bn3_kernel_revised.close()
                    else:
                        if gpu == 0:
                            continue
                        os.symlink(f'../GPU_0/{kernel_0}', f'{kernels_gpu_path}/{kernel_0}')
