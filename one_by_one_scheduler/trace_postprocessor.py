import os
import re
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

def parse_custom_call(kernel_string):
  #s = kernel_string.find('call.')
  #custom_call_num = kernel_string[s+5:s+8]
  #return int(custom_call_num)
  return int(kernel_string.split('.')[1].split('_')[0])


EXP_NAME = f'packet_{args.packet_size}_buffer_{args.buffer}_gpu_{args.gpu}_sync_{1 if args.sync else 0}_simd_{args.simd}'
BASE_PATH = f'/home/jueonpark/tracegen/traces/{args.model}/traces/{EXP_NAME}'
if os.path.exists(BASE_PATH):
  raise Exception(f'Please check the path {BASE_PATH}')
else:
  os.mkdir(BASE_PATH)

NDP_TRACE_PATH = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/{EXP_NAME}'

kernelslist_file = open(f'/home/jueonpark/tracegen/traces/{args.model}/kernelslist.g', 'r')
kernelslist_tmp_file = open(f'/home/jueonpark/tracegen/traces/{args.model}/traces/{EXP_NAME}/kernelslist.g.tmp', 'w')

kernel_list = kernelslist_file.read().split('\n\n')
if kernel_list[-1] == "":
  kernel_list = kernel_list[:-1]
for ndp_gpu_kernels in tqdm(kernel_list):
    custom_call_id = -1
    page_table = False
    no_cxl = False
    kernels = ndp_gpu_kernels.split('\n')
    tmp_on_the_fly_traceg = ""
    is_pattern_matching_on_the_fly = False
    for kernel in kernels:
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
      elif '_ON_THE_FLY' in kernel:
        # case for non-pattern-matching on-the-fly automatically generated from compiler
        tmp_on_the_fly_traceg = kernel
        kernelslist_tmp_file.write(kernel + '\n')
        continue
      if 'kernel-' in kernel: # GPU kernel
        input_file_path = f'/home/jueonpark/tracegen/traces/{args.model}/traces/{kernel}'
        output_file_path = f'/home/jueonpark/tracegen/traces/{args.model}/traces/{EXP_NAME}/{kernel}_post.traceg'
        f = open(input_file_path, 'r')
        kernelslist_tmp_file.write(kernel + '_post.traceg\n')
        output = open(output_file_path, 'w')
        if page_table == True:
          # for on-the-fly write, usually for ph3
          page_table_file = open(f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/{EXP_NAME}/page_table_header_custom-call.{custom_call_id}.traceg', 'r')
          output.write(page_table_file.read() + '\n')
          page_table_file.close()
        if not no_cxl: # meaning on-the-fly read
          if is_pattern_matching_on_the_fly:
            # original script for pattern matching
            for line in f.readlines():
              if 'STG' in line:
                for word in line.split():
                  if "0x" in word:
                    line = re.sub('0x7', '0x1007', line)
              output.write(line)
            output.close()
          # end of original script
          else:
            print(kernel)
            print(tmp_on_the_fly_traceg)
            on_the_fly_addr = ""
            # open tmp_on_the_fly_traceg file and get the target address
            try:
              tmp_on_the_fly_traceg = open(f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/{EXP_NAME}/{tmp_on_the_fly_traceg}', 'r').read()
              on_the_fly_addr = tmp_on_the_fly_traceg.split("SET_FILTER ")[1].split(" ", 1)[0].split("0x100")[1]
            except:
              on_the_fly_addr = "no on-the-fly, continuing..."
            print(on_the_fly_addr)
            tmp_line_list = []
            rewritten_last_stg = False
            for line in reversed(f.readlines()):
              if ('STG' in line) and (on_the_fly_addr in line) and (not rewritten_last_stg):
                rewritten_last_stg = True
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