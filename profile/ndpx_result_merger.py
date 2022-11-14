# input: ndpx_scheduling_table.csv
# things to rewrite: NdpxOpLayer, GpuKernelLayer
import os
import pathlib
import argparse
from xla_metadata_parser import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)

# The content consists of:
# NdpxKernel information
# - NdpxKernel
# - original instruction name
# - NdpxKernel's shape size
# - #input
# - #output
# - #op
# - estimated cost of NdpxKernel
# - layer(metadata) of NdpxKernel
# overlapping GPU kernel information
# - GpuKernel (basically GPU instruction's name)
# - pre-measured GpuKernel's cost
# - layer of GpuKernel
# execution option
# - on-the-fly or not (true if on-the-fly)
if __name__ == "__main__":
  args = parser.parse_args()
  exp_path = os.getenv("EXP_PATH")
  model = ""
  if (args.model).find("bert") != -1:
    model = "bert"
  elif (args.model).find("resnet") != -1:
    model = "resnet"
  elif (args.model).find("mobilenet") != -1:
    model = "mobilenet"
  elif (args.model).find("transformer") != -1:
    model = "transformer"
  elif (args.model).find("dlrm") != -1:
    model = "dlrm"
  elif (args.model).find("vit") != -1:
    model = "vit"
  elif (args.model).find("transformer") != -1:
    model = "transformer"
  else:
    exit(0)

  total_result = open(f'/home/jueonpark/tracegen/csv_files/{args.model}-NDPX_baseline_64-1-nosync-ndpx-cycle.csv', 'r').read()
  total_results = total_result.split('\n')
  kernelslist = open(f'/home/jueonpark/tracegen/traces/{args.model}/kernelslist.g', 'r').read()
  kernelslists = kernelslist.split('\n\n')

  xla_hlo_path_str = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  table_paths = list(xla_hlo_path.glob("*ndpx_scheduling_table*"))
  original_data = open(table_paths[0], "r").read().split("\n")[0] + "\n"
  for table_path in table_paths:
    table = open(table_path, "r").read()
    original_data += table.split("\n", 1)[1]
  original_results = original_data.split("\n")
  
  output_path = f'/home/jueonpark/tracegen/experiments_results/{args.model}/ndpx_estimation_result.csv'
  output = open(output_path, "w+")

  new_header = "NdpxKernel,OverlappedKernelNum,NdpxKernelDimm,#inputs,#outputs,#ops,EstimatedNdpxCost,RealNdpxCost,MultipleNdpx,IntendedSchedule\n"
  output.write(new_header)
  for original_row in original_results[1:-1]:
    original_elements = original_row.split(",")
    # find kernel number and real cycle
    found = False
    multiple_ndpx = False
    intended_schedule = False
    kernel_num = ""
    kernel_cycle = 0
    for kernel in kernelslists:
      # Get the overlapping information from kernelslist.g
      # Thunk: custom-call.6:__cublas$gemm
      # Kernel Name: turing_fp16_s884gemm_fp16_64x64_ldg8_f2f_nn
      # print(kernel)
      knl_elements = kernel.split('\n', 4)
      if len(knl_elements) < 4:
        continue
      if knl_elements[3].find(original_elements[0]) != -1:
        found = True
        kernel_num = kernel.split('\n')[-1]
        kernel_num = kernel_num.split('-')[1]
        kernel_num = kernel_num.split('.')[0]
        # find whether there are multiple ndpx overlapped
        if knl_elements[4].count('_NDP_') > 1:
          multiple_ndpx = True
        # find whether the scheduling is done as intended
        candidate = original_elements[0].rsplit('$', 1)[1].split('.traceg')[0]
        if knl_elements[0].find(candidate) != -1 and knl_elements[4].find(original_elements[0]) != -1:
          intended_schedule = True
        break
    if found:
      # find from total_results
      for tr_row in total_results:
        if tr_row.find(kernel_num) != -1 and tr_row.find('NDP_OP') != -1:
          kernel_cycle = tr_row.split(',')[5]
          break

    new_results = ""
    original_elements = original_row.split(",")

    new_results = original_elements[0] + "," + \
                  kernel_num + "," + \
                  original_elements[2] + "," + \
                  original_elements[3] + "," + \
                  original_elements[4] + "," + \
                  original_elements[5] + "," + \
                  original_elements[6] + "," + \
                  str(kernel_cycle) + ',' + \
                  str(multiple_ndpx) + ',' + \
                  str(intended_schedule) + "\n"
    output.write(new_results)
