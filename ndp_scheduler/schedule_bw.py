import os
import argparse
from utils import parse_schedule
from utils import dump_schedule
from utils import type_kernel_bw

parser = argparse.ArgumentParser()
parser.add_argument('--schedule', type=str, required=True)
# path for gpu 4 buffer 4
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

def check_inv(compute_kernels_corrected):
    for i, (type_, _) in enumerate(compute_kernels_corrected[:-1]):
        if i % 2 == 0:
            assert('dgrad' in type_)
        else:
            assert('wgrad' in type_)
    assert('wgrad' in compute_kernels_corrected[-1][0])

with open(args.schedule, 'r') as f:
    schedule = parse_schedule(f.read())

compute_kernels = []
memory_kernels = []
for kernel in schedule:
    kernel_type = type_kernel_bw(kernel['kernel_name'])
    if kernel_type is not None:
        compute_kernels.append((kernel_type, kernel))
    else:
        memory_kernels.append(kernel)

for kernel in memory_kernels:
    kernel['no_cxl'] = True

compute_kernels_corrected = []
for i, (type_, kernel) in enumerate(compute_kernels):
    if type_ == "conv" or type_ == "gemm":
        for j, (type__, kernel_) in enumerate(compute_kernels[i:i+10]):
            print(f'{j}: {type__}, {kernel_["kernel_name"]}')
        yes_or_no = input(f"Maybe {kernel['kernel_name']} dgrad? (y/n): ")
        print('\n')
        if yes_or_no == 'y' or yes_or_no == 'Y':
            yes_or_no = True
            compute_kernels_corrected.append((type_ + '_dgrad', kernel))
        elif yes_or_no == 'n' or yes_or_no == 'N':
            yes_or_no = False
            compute_kernels_corrected.append((type_ + '_wgrad', kernel))
        else:
            raise Exception('Y/N or y/n')
    else:
        compute_kernels_corrected.append((type_, kernel))

# reposition wgemm
compute_kernels_corrected = [compute_kernels_corrected[0]] + [compute_kernels_corrected[-1]] + compute_kernels_corrected[1:-1]

print("\n\nResults:")
for i, (type_, kernel) in enumerate(compute_kernels_corrected):
    print(f'{i}: {type_}, {kernel["kernel_name"]}')

#check_inv(compute_kernels_corrected)

# NO_CXL for all kernel except for dgrad
for type_, kernel in compute_kernels_corrected:
    if not 'dgrad' in type_:
        kernel['no_cxl'] = True

print(dump_schedule([i for _, i in compute_kernels_corrected]))

# construct NDP kernels
ndp_kernels = []
ndp_kernel_files = os.listdir(args.path)
dgrads = [kernel for type_, kernel in compute_kernels_corrected if 'dgrad' in type_]
wgrads = [kernel for type_, kernel in compute_kernels_corrected if 'wgrad' in type_]
for kernel in wgrads:
    custom_call_string = kernel['thunk_name'].split(':')[0]
    custom_call_num = custom_call_string[custom_call_string.rfind('_')+1:]
    ndp_kernels.append(sorted([kernel_file for kernel_file in ndp_kernel_files if f'.{custom_call_num}' in kernel_file]))

# construct optimizer
optim_files = []
for kernel in wgrads:
    optim_files_kernel = []
    for optim in kernel['optims']:
        optim_files_kernel += [kernel_file for kernel_file in ndp_kernel_files if optim.split(':')[0] in kernel_file]
    optim_files.append(sorted(optim_files_kernel))

ndp_optim_files = []
for i in range(len(optim_files)):
    if i == len(optim_files) - 1:
        ndp_optim_files.append(optim_files[i])
    else:
        ndp_optim_files.append(ndp_kernels[i+1])
        ndp_optim_files.append(optim_files[i])
ndp_optim_files = [[]] + ndp_optim_files[:-1]
print(ndp_optim_files, len(ndp_optim_files), len(compute_kernels_corrected))

for ndp_kernels, (type_, kernel) in zip(ndp_optim_files, compute_kernels_corrected):
    if 'dgrad' in type_ or ndp_optim_files.index(ndp_kernels) == len(ndp_optim_files) - 1:
        optims = [optim for optim in ndp_kernels if 'reduce' in optim]
        optims.append('_BAR_')
        optims += [optim for optim in ndp_kernels if not 'reduce' in optim]
        kernel['ndp_kernels'] = optims
    elif 'wgrad' in type_:
        ordered = dict()
        for ndp_kernel in ndp_kernels:
            if 'max_pool.' in ndp_kernel:
                ordered['max_pool'] = ndp_kernel
            elif 'ph1.' in ndp_kernel:
                ordered['ph1'] = ndp_kernel
            elif 'ph2.' in ndp_kernel:
                ordered['ph2'] = ndp_kernel
            elif 'ph3.' in ndp_kernel:
                ordered['ph3'] = ndp_kernel
            elif 'bias.' in ndp_kernel:
                ordered['bias'] = ndp_kernel
            elif 'bias_reduce.' in ndp_kernel:
                ordered['bias_reduce'] = ndp_kernel
            else:
                print(f'Unknown ndp kernel: {ndp_kernel}')
        kernel['ndp_kernels'] = []
        if 'max_pool' in ordered:
            kernel['ndp_kernels'].append(ordered['max_pool'])
        if 'ph1' in ordered:
            kernel['ndp_kernels'].append('_BAR_')
            kernel['ndp_kernels'].append(ordered['ph1'])
        if 'ph2' in ordered:
            kernel['ndp_kernels'].append('_BAR_')
            kernel['ndp_kernels'].append(ordered['ph2'])
        if 'ph3' in ordered:
            kernel['ndp_kernels'].append('_BAR_')
            kernel['ndp_kernels'].append(ordered['ph3'])
        if 'bias' in ordered:
            kernel['ndp_kernels'].append('_BAR_')
            kernel['ndp_kernels'].append(ordered['bias'])
        if 'bias_reduce' in ordered:
            kernel['ndp_kernels'].append('_BAR_')
            kernel['ndp_kernels'].append(ordered['bias_reduce'])
    else:
        raise Exception('type is invalid')

scheduling = dump_schedule(memory_kernels) + dump_schedule([kernel for _, kernel in compute_kernels_corrected])

with open(args.output, 'w') as f:
    f.write(scheduling)
