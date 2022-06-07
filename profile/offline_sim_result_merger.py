# merges offline_execution_result.csv and simulation results

# input: gpu_kernel_estimation_table_cluster_0.csv
# merges estimated gpu cycles and real gpu cycles
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fw', type=str, help="forward.csv", required=True)
parser.add_argument('-b', '--bw', type=str, help="backward.csv", required=True)
parser.add_argument('--kfw', type=str, help="kernelslist.g.fw", required=True)
parser.add_argument('--kbw', type=str, help="kernelslist.g.bw", required=True)
parser.add_argument('-o', '--of', type=str, help="offline_execution_result.csv", required=True)

if __name__ == "__main__":
  args = parser.parse_args()

  # first merge fw and bw
  total_result = open(args.fw, 'r').read()
  total_result += open(args.bw, 'r').read().split("\n", 1)[1]
  overall_file = open('output.csv', 'w')
  overall_file.write(total_result)

  # we then merge the estimation
  # - add the real result column to the estimation file
  # read gpu kernel name and find the appropriate kernel
  kernelslist = open(args.kfw, 'r').read() + '\n\n' + open(args.kbw, 'r').read()
  kernelslists = kernelslist.split('\n\n')
  off_file_object = csv.reader(open(args.of, 'r'))
  off_output = open("offline_execution_result.csv", "w+")
  
  new_header = "GpuKernel,SimulationCycle\n"
  off_output.write(new_header)
  estimation_result = []
  estimation_result.append(new_header)
  prev_header = next(off_file_object)
  
  total_results = total_result.split('\n')
  for off_row in off_file_object:
    # find kernel number and real cycle
    found = False
    kernel_name = ""
    kernel_cycle = 0
    for kernel in kernelslists:
      # Thunk: custom-call.6:__cublas$gemm
      # Kernel Name: turing_fp16_s884gemm_fp16_64x64_ldg8_f2f_nn
      knl_elements = kernel.split('\n')
      if knl_elements[0].find(off_row[0]) != -1:
        found = True
        kernel_name = knl_elements[1].split(": ")[1]
        print(kernel_name)
        break
    if found:
      # find from total_results
      for tr_row in total_results:
        if tr_row.find(kernel_name) != -1:
          print(tr_row)
          kernel_cycle = tr_row.split(',')[6]
          break
    # off_row[1] + ',' + \
    output_line = off_row[0] + ',' + str(kernel_cycle) + "\n"
    off_output.write(output_line)