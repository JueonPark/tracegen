# input: ndpx_scheduling_table_cluster_0.csv
# merges estimated ndpx cycles and real ndpx cycles
import os
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, help="full-cycle.csv", required=True)
parser.add_argument('--kfw', type=str, help="kernelslist.g.fw", required=True)
parser.add_argument('--kbw', type=str, help="kernelslist.g.bw", required=True)
parser.add_argument('-n', '--ne', type=str, help="ndpx_estimation.csv", required=True)

if __name__ == "__main__":
  args = parser.parse_args()
  exp_path = os.getenv("EXP_PATH")

  # full cycle information
  total_result = open(args.csv, 'r').read()
  # NDPX trace file path
  model = exp_path.split('/')[-1]
  # ndpx_scheduling_table_cluster_0.csv
  ndpx_trace_dir_path = f'/home/jueonpark/tracegen/traces/{model}/xla_hlo/packet_32_buffer_1_gpu_1_sync_0_simd_8/'
  ndpx_trace_files = os.listdir(ndpx_trace_dir_path)
  for ndpx_trace_file in ndpx_trace_files:
    if 'page_table' in ndpx_trace_file:
      ndpx_trace_files.remove(ndpx_trace_file)
  print(ndpx_trace_files)
  
  # then merge the estimation
  # - add the real result column to the estimation file
  # read gpu kernel name and find the appropriate kernel
  kernelslist = open(args.kfw, 'r').read() + '\n\n' + open(args.kbw, 'r').read()
  kernelslists = kernelslist.split('\n\n')
  ne_file_object = csv.reader(open(args.ne, 'r'))
  ne_output_path = os.path.join(exp_path, "ndpx_estimation_result.csv")
  ne_output = open(ne_output_path, "w+")
  
  # The header of ndpx_scheduling_table_cluster_0.csv consists of:
  # - NdpxKernel,
  # - NdpxKernelDimm,
  # - #inputs,
  # - #outputs,
  # - #ops,
  # - EstimatedNdpxCost,
  # - NdpxOpLayer,
  # - GpuKernel,
  # - EstimatedGpuCost,
  # - GpuKernelLayer,
  # - OnTheFly
  new_header = "NdpxKernel,OverlappedKernelNum,NdpxKernelDimm,#inputs,#outputs,#ops,EstimatedNdpxCost,RealNdpxCost,MultipleNdpx,IntendedSchedule\n"
  ne_output.write(new_header)
  estimation_result = []
  estimation_result.append(new_header)
  prev_header = next(ne_file_object)
  
  total_results = total_result.split('\n')
  for ne_row in ne_file_object:
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
      if knl_elements[4].find(ne_row[0]) != -1:
        found = True
        kernel_num = kernel.split('\n')[-1]
        kernel_num = kernel_num.split('-')[1]
        kernel_num = kernel_num.split('.')[0]
        # find whether there are multiple ndpx overlapped
        if knl_elements[4].count('_NDP_') > 1:
          multiple_ndpx = True
        # find whether the scheduling is done as intended
        candidate = ne_row[0].rsplit('$', 1)[1].split('.traceg')[0]
        if knl_elements[0].find(candidate) != -1 and knl_elements[4].find(ne_row[0]) != -1:
          intended_schedule = True
        break
    if found:
      # find from total_results
      for tr_row in total_results:
        if tr_row.find(kernel_num) != -1 and tr_row.find('NDP_OP') != -1:
          kernel_cycle = tr_row.split(',')[6]
          break
    # output format would be:
    # - NdpxKernel
    # - OverlappedKernelNum
    # - ShapeSize
    # - #input
    # - #output
    # - #op
    # - EstimateNdpxCost
    # - RealNdpxCost
    # - MultipleNdpx
    # - IntendedSchedule
    output_line = ne_row[0] + ',' + \
                  kernel_num + ',' + \
                  ne_row[1] + ',' + \
                  ne_row[2] + ',' + \
                  ne_row[3] + ',' + \
                  ne_row[4] + ',' + \
                  ne_row[5] + ',' + \
                  str(kernel_cycle) + ',' + \
                  str(multiple_ndpx) + ',' + \
                  str(intended_schedule) + "\n"
    ne_output.write(output_line)