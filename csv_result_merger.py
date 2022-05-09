import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fw', type=str, help="forward.csv", required=True)
parser.add_argument('-b', '--bw', type=str, help="backward.csv", required=True)
parser.add_argument('--kfw', type=str, help="kernelslist.g.fw", required=True)
parser.add_argument('--kbw', type=str, help="kernelslist.g.bw", required=True)
parser.add_argument('-g', '--ge', type=str, help="gpu_estimation.csv", required=True)

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
  ge_file_object = csv.reader(open(args.ge, 'r'))
  ge_output = open("estimation_result.csv", "w+")
  
  new_header = "GpuKernel,KernelName,ShapeSize,InputSize,NdpxInput,#Ops,EstimatedCost,RealCycles"
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
          kernel_cycle = tr_row.split(',')[6]
          break
    output_line = ge_row[0] + ',' + \
                  kernel_name + ',' + \
                  ge_row[1] + ',' + \
                  ge_row[2] + ',' + \
                  ge_row[3] + ',' + \
                  ge_row[4] + ',' + \
                  ge_row[5] + ',' + \
                  str(kernel_cycle) + "\n"
    ge_output.write(output_line)