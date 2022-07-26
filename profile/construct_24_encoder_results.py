import sys
import os
import re
import argparse

from hlo_dependency_parser import HloDepdendencyManager

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True, help="result csv file (.csv)")
parser.add_argument('--hlo', type=str, required=True, help="xla hlo graph path (.mlir)")
parser.add_argument('--kfw', type=str, help="kernelslist.g.fw", required=True)
parser.add_argument('--kbw', type=str, help="kernelslist.g.bw", required=True)

if __name__ == "__main__":
  args = parser.parse_args()

  # generate *full_bert_large_breakdown.csv
  fw_kernelslist = open(args.kfw, 'r').read()
  bw_kernelslist = open(args.kbw, 'r').read()
  hlo_manager = HloDepdendencyManager(open(args.hlo).read())
  
  # forward layer 1
  fw_layer_1_list = []
  fw_layer_1_start_found = False
  fw_layer_1_end_found = False
  fw_layer_1_start = 99999999
  fw_layer_1_end = -1

  # forward layer 1
  fw_kernel_nums = []
  for kernel in fw_kernelslist.split("\n\n"):
    kernel_elemenets = kernel.split("\n")
    kernel_instr = ""
    try:
      kernel_instr = kernel_elemenets[0].split("// Thunk: ")[1]
    except:
      print(kernel_elemenets)
      break
    if "__cublas$gemm" in kernel_instr:
      kernel_instr = kernel_instr.split(":__cublas$gemm")[0]
    kernel_num = int(kernel_elemenets[-1].split("kernel-")[1].split(".traceg")[0])
    fw_kernel_nums.append(kernel_num)
    # # find fw_layer_1_start
    # if not fw_layer_1_start_found:
    #   if kernel_instr in hlo_manager.metadata_table:
    #     if re.search("model+", hlo_manager.metadata_table[kernel_instr]) \
    #           and re.search("layer_1+", hlo_manager.metadata_table[kernel_instr]):
    #       print('fw_layer_1_start_found')
    #       fw_layer_1_start = int(kernel_num)
    #       fw_layer_1_start_found = True
    # # find fw_layer_1_end
    # if fw_layer_1_start_found and not fw_layer_1_end_found:
    #   if kernel_instr in hlo_manager.metadata_table:
    #     if re.search("model+", hlo_manager.metadata_table[kernel_instr]) \
    #           and re.search("layer_2+", hlo_manager.metadata_table[kernel_instr]):
    #       print('fw_layer_1_end_found')
    #       fw_layer_1_end = int(kernel_num) - 1
    #       fw_layer_1_end_found = True
    
    # append kernel number
    if kernel_instr in hlo_manager.metadata_table:
      if re.search("model+", hlo_manager.metadata_table[kernel_instr]) \
            and re.search("layer_1+", hlo_manager.metadata_table[kernel_instr]) \
            and not re.search("gradient_tape+", hlo_manager.metadata_table[kernel_instr]):
        fw_layer_1_list.append(kernel_num)
  
  fw_layer_1_start = min(fw_layer_1_list)
  fw_layer_1_end = max(fw_layer_1_list)
  print(fw_layer_1_start)
  print(fw_layer_1_end)
  # finally, add all the kernel number beween fw_layer_1_start and fw_layer_1_end
  for i in range(fw_layer_1_start, fw_layer_1_end, 1):
    if i not in fw_layer_1_list and i in fw_kernel_nums:
      fw_layer_1_list.append(i)

  print(fw_layer_1_list)

  #############################################################################

  # backward layer 1
  bw_layer_1_list = []
  bw_layer_1_start_found = False
  bw_layer_1_end_found = False
  bw_layer_1_start = 99999999
  bw_layer_1_end = -1

  # backward layer 1
  bw_kernel_nums = []
  for kernel in bw_kernelslist.split("\n\n"):
    kernel_elemenets = kernel.split("\n")
    kernel_instr = ""
    try:
      kernel_instr = kernel_elemenets[0].split("// Thunk: ")[1]
    except:
      print(kernel_elemenets)
      break
    if "__cublas$gemm" in kernel_instr:
      kernel_instr = kernel_instr.split(":__cublas$gemm")[0]
    kernel_num = int(kernel_elemenets[-1].split("kernel-")[1].split(".traceg")[0])
    bw_kernel_nums.append(kernel_num)
    # # find bw_layer_1_start
    # if not bw_layer_1_start_found:
    #   if kernel_instr in hlo_manager.metadata_table:
    #     if re.search("gradient_tape+", hlo_manager.metadata_table[kernel_instr]) \
    #           and re.search("layer_1+", hlo_manager.metadata_table[kernel_instr]):
    #       print('bw_layer_1_start_found')
    #       bw_layer_1_start = int(kernel_num)
    #       bw_layer_1_start_found = True
    # # find bw_layer_1_end
    # if bw_layer_1_start_found and not bw_layer_1_end_found:
    #   if kernel_instr in hlo_manager.metadata_table:
    #     if re.search("gradient_tape+", hlo_manager.metadata_table[kernel_instr]) \
    #           and re.search("layer_0+", hlo_manager.metadata_table[kernel_instr]):
    #       print('bw_layer_1_end_found')
    #       bw_layer_1_end = int(kernel_num) - 1
    #       bw_layer_1_end_found = True
    
    # append kernel number
    if kernel_instr in hlo_manager.metadata_table:
      if re.search("gradient_tape+", hlo_manager.metadata_table[kernel_instr]) \
            and re.search("layer_1+", hlo_manager.metadata_table[kernel_instr]):
        bw_layer_1_list.append(kernel_num)
  
  bw_layer_1_start = min(bw_layer_1_list)
  bw_layer_1_end = max(bw_layer_1_list)
  print(bw_layer_1_start)
  print(bw_layer_1_end)
  # finally, add all the kernel number beween fw_layer_1_start and fw_layer_1_end
  for i in range(bw_layer_1_start, bw_layer_1_end, 1):
    if i not in bw_layer_1_list and i in bw_kernel_nums:
      bw_layer_1_list.append(i)

  print(bw_layer_1_list)

  # after finding layer_1, add the layer_1 to full_bert_large_breakdown.csv file.
  exp_path = os.getenv("EXP_PATH")
  output_name = (args.csv).split("breakdown.csv")[0] + "24layer_breakdown.csv"
  output_path = os.path.join(exp_path, output_name)
  output = open(output_path, "w+")

  original_result = open(args.csv, "r").read()
  new_result = original_result
  rows_to_add = []
  # find rows to add (basically layer 1)
  for original_row in original_result.split("\n")[1:]:
    row_elements = original_row.split(",")
    if (len(row_elements) < 4):
      continue
    if int(row_elements[3]) in fw_layer_1_list:
      rows_to_add.append(original_row + "\n")
    elif int(row_elements[3]) in bw_layer_1_list:
      rows_to_add.append(original_row + "\n")
  # add layer_1 for 22 times
  for i in range(22):
    for row in rows_to_add:
      new_result += row
  
  output.write(new_result)