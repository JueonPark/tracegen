# input: gpu_kernel_estimation_table_cluster_0.csv
# merges estimated gpu cycles and real gpu cycles
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--csv', type=str, help="total_result.csv", required=True)
parser.add_argument('--kfw', type=str, help="kernelslist.g.fw", required=True)
parser.add_argument('--kbw', type=str, help="kernelslist.g.bw", required=True)
parser.add_argument('-g', '--ge', type=str, help="gpu_estimation.csv", required=True)

def kernel_name_rewriter(kernel_name):
  if kernel_name.find("turing_fp16") != -1:
    if kernel_name.find("_nn") != -1:
      return "GEMM_nn"
    elif kernel_name.find("_nt") != -1:
      return "GEMM_nt"
    elif kernel_name.find("_tn") != -1:
      return "GEMM_tn"
    elif kernel_name.find("_tt") != -1:
      return "GEMM_tt"
  else:
    return kernel_name.split("_")[0]

if __name__ == "__main__":
  args = parser.parse_args()

  # we first merge fw and bw
  total_result = open(args.csv, 'r').read()
  overall_file = open('output.csv', 'w')
  overall_file.write(total_result)

  # we then merge the estimation
  # - add the real result column to the estimation file
  # read gpu kernel name and find the appropriate kernel
  kernelslist = open(args.kfw, 'r').read() + '\n\n' + open(args.kbw, 'r').read()
  kernelslists = kernelslist.split('\n\n')
  ge_file_object = csv.reader(open(args.ge, 'r'))
  ge_output = open("gpu_estimation_result.csv", "w+")
  
  new_header = "GpuKernel,KernelName,ShapeSize,InputSize,NdpxInput,#Ops,EstimatedCost,RealCycles\n"
  ge_output.write(new_header)
  estimation_result = []
  estimation_result.append(new_header)
  prev_header = next(ge_file_object)
  
  total_results = total_result.split('\n')
  for ge_row in ge_file_object:
    # find kernel number and real cycle
    found = False
    kernel_name = ""
    kernel_cycle = 0
    for kernel in kernelslists:
      # Thunk: custom-call.6:__cublas$gemm
      # Kernel Name: turing_fp16_s884gemm_fp16_64x64_ldg8_f2f_nn
      knl_elements = kernel.split('\n')
      if knl_elements[0].find(ge_row[0]) != -1:
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
    output_line = ge_row[0] + ',' + \
                  kernel_name_rewriter(kernel_name) + ',' + \
                  ge_row[1] + ',' + \
                  ge_row[2] + ',' + \
                  ge_row[3] + ',' + \
                  ge_row[4] + ',' + \
                  ge_row[5] + ',' + \
                  str(kernel_cycle) + "\n"
    ge_output.write(output_line)