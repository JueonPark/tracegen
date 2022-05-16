# input: ndpx_scheduling_table_cluster_0.csv
# merges estimated ndpx cycles and real ndpx cycles
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fw', type=str, help="fw-full-cycle.csv", required=True)
parser.add_argument('--bw', type=str, help="bw-full-cycle.csv", required=True)
parser.add_argument('--kfw', type=str, help="kernelslist.g.fw", required=True)
parser.add_argument('--kbw', type=str, help="kernelslist.g.bw", required=True)
parser.add_argument('-n', '--ne', type=str, help="ndpx_estimation.csv", required=True)

if __name__ == "__main__":
  args = parser.parse_args()

  # we first merge fw and bw
  total_result = open(args.fw, 'r').read()
  total_result += open(args.bw, 'r').read().split("\n", 1)[1]
  overall_file = open('output.csv', 'w')
  overall_file.write(total_result)

  # we then merge the estimation
  # - add the real result column to the estimation file
  # read gpu kernel name and find the appropriate kernel
  kernelslist = open(args.kfw, 'r').read() + '\n\n' + open(args.kbw, 'r').read()
  kernelslists = kernelslist.split('\n\n')
  ne_file_object = csv.reader(open(args.ne, 'r'))
  ne_output = open("ndpx_estimation_result.csv", "w+")
  
  new_header = "NdpxKernel,OverlappedKernelNum,NdpxKernelDimm,#inputs,#outputs,#ops,EstimatedNdpxCost,RealNdpxCost,MultipleNdpx\n"
  ne_output.write(new_header)
  estimation_result = []
  estimation_result.append(new_header)
  prev_header = next(ne_file_object)
  
  total_results = total_result.split('\n')
  for ne_row in ne_file_object:
    # find kernel number and real cycle
    found = False
    multiple_ndpx = False
    kernel_num = ""
    kernel_cycle = 0
    for kernel in kernelslists:
      # Thunk: custom-call.6:__cublas$gemm
      # Kernel Name: turing_fp16_s884gemm_fp16_64x64_ldg8_f2f_nn
      knl_elements = kernel.split('\n', 4)
      if knl_elements[4].find(ne_row[0]) != -1:
        found = True
        kernel_num = kernel.split('\n')[-1]
        kernel_num = kernel_num.split('-')[1]
        kernel_num = kernel_num.split('.')[0]
        print(kernel_num)
        if knl_elements[4].count('_NDP_') > 1:
          multiple_ndpx = True
        break
    if found:
      # find from total_results
      for tr_row in total_results:
        if tr_row.find(kernel_num) != -1 and tr_row.find('NDP_OP') != -1:
          print(tr_row)
          kernel_cycle = tr_row.split(',')[6]
          break
    output_line = ""
    if multiple_ndpx:
      output_line = ne_row[0] + ',' + \
                    kernel_num + ',' + \
                    ne_row[1] + ',' + \
                    ne_row[3] + ',' + \
                    ne_row[4] + ',' + \
                    ne_row[5] + ',' + \
                    ne_row[2] + ',' + \
                    str(kernel_cycle) + ',' + \
                    "True\n"
    else:
      output_line = ne_row[0] + ',' + \
                    kernel_num + ',' + \
                    ne_row[1] + ',' + \
                    ne_row[3] + ',' + \
                    ne_row[4] + ',' + \
                    ne_row[5] + ',' + \
                    ne_row[2] + ',' + \
                    str(kernel_cycle) + ',' + \
                    "False\n"
    ne_output.write(output_line)