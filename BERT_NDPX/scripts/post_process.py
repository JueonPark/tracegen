import os
import re
import argparse
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--packet-size', type=int, required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--buffer', type=int, required=True)
parser.add_argument('--sync', type=bool, default=False, required=False)
parser.add_argument('--simd', type=int, required=True)
parser.add_argument('--passes', type=str, required=True) # fw, bw or all
args = parser.parse_args()

def parse_custom_call(kernel_string):
    #s = kernel_string.find('call.')
    #custom_call_num = kernel_string[s+5:s+8]
    #return int(custom_call_num)
    return int(kernel_string.split('.')[1].split('_')[0])

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

BASE_PATHS = [f'/home/jueonpark/tracegen/traces/{args.model}/traces_{p}/{EXP_NAME}_{p}' for p in passes]
for base_path in BASE_PATHS:
    if os.path.exists(base_path):
        raise Exception(f'Please check the path {base_path}')
    else:
        os.mkdir(base_path)

NDP_TRACE_PATHS = [f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo_{p}/{EXP_NAME}' for p in passes]

kernelslist_files = [open(f'/home/jueonpark/tracegen/traces/{args.model}/traces_{p}/kernelslist.g.{p}', 'r') for p in passes]
kernelslist_tmp_files = [open(f'/home/jueonpark/tracegen/traces/{args.model}/traces_{p}/{EXP_NAME}_{p}/kernelslist.g.tmp.{p}', 'w') for p in passes]

for p, kernelslist_file, kernelslist_tmp_file in zip(passes, kernelslist_files, kernelslist_tmp_files):
    kernel_list = kernelslist_file.read().split('\n\n')
    if kernel_list[-1] == "":
        kernel_list = kernel_list[:-1]
    for ndp_gpu_kernels in tqdm(kernel_list):
        custom_call_id = -1
        page_table = False
        no_cxl = False
        kernels = ndp_gpu_kernels.split('\n')
        tmp_on_the_fly_traceg = ""
        for kernel in kernels:
            # print(kernel)
            if '//' in kernel:
                continue
            if 'NO_CXL' in kernel: # write data to gpu memory, not thru CXL
                no_cxl = True
                continue
            if '_BAR_' in kernel: # barrier
                kernelslist_tmp_file.write(kernel + '\n')
                continue
            if '_NDP' in kernel: # not-on-the-fly
                kernelslist_tmp_file.write(kernel + '\n')
                continue
            elif '_ON_THE_FLY' in kernel and 'ph3' in kernel or 'PAGE_TABLE' in kernel:
                page_table = True
                custom_call_id = parse_custom_call(kernel)
                kernelslist_tmp_file.write(kernel + '\n')
                continue
            elif '_ON_THE_FLY' in kernel and 'ph1' in kernel:
                tmp_on_the_fly_traceg = kernel
                kernelslist_tmp_file.write(kernel + '\n')
                continue
            if 'kernel-' in kernel: # GPU kernel
                input_file_path = f'/home/jueonpark/tracegen/traces/{args.model}/traces_{p}/{kernel}'
                output_file_path = f'/home/jueonpark/tracegen/traces/{args.model}/traces_{p}/{EXP_NAME}_{p}/{kernel}_post.traceg'
                f = open(input_file_path, 'r')
                kernelslist_tmp_file.write(kernel + '_post.traceg\n')
                output = open(output_file_path, 'w')
                if page_table == True:
                    page_table_file = open(f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo_{p}/{EXP_NAME}/page_table_header_custom-call.{custom_call_id}.traceg', 'r')
                    output.write(page_table_file.read() + '\n')
                    page_table_file.close()
                if not no_cxl:
                  # handled 
                  print(kernel)
                  print(tmp_on_the_fly_traceg)
                  on_the_fly_addr = ""
                  # open tmp_on_the_fly_traceg file and get the target address
                  try:
                    tmp_on_the_fly_traceg = open(f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo_{p}/{EXP_NAME}/{tmp_on_the_fly_traceg}', 'r').read()
                    on_the_fly_addr = tmp_on_the_fly_traceg.split("SET_FILTER ")[1].split(" ", 1)[0].split("0x100")[1]
                  except:
                    on_the_fly_addr = "no on-the-fly, continuing..."
                  print(on_the_fly_addr)
                  tmp_line_list = []
                  last_stg_detected = False
                  for line in reversed(f.readlines()):
                    if ('STG' in line) and (on_the_fly_addr in line) and (not last_stg_detected):
                      last_stg_detected = True
                      for word in line.split():
                        if "0x" in word:
                          line = re.sub('0x7', '0x1007', line)
                    tmp_line_list.append(line)
                  for line in reversed(tmp_line_list):
                    output.write(line)
                  output.close()
                else:
                  output.write(f.read())
                output.close()
                f.close()
        kernelslist_tmp_file.write('\n')
    kernelslist_file.close()
    kernelslist_tmp_file.close()

