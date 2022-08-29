# merges offline_execution_result.csv and simulation results

# input: gpu_kernel_estimation_table_cluster_0.csv
# merges estimated gpu cycles and real gpu cycles
import csv
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="model", required=True)

if __name__ == "__main__":
  args = parser.parse_args()
  csv_path = f'/home/jueonpark/tracegen/csv_files/{args.model}-NDPX_baseline_64-1-nosync.csv'

  total_result = open(csv_path, 'r').read()

  kernelslist_path = f'/home/jueonpark/tracegen/traces/{args.model}/kernelslist.g'
  kernelslist = open(kernelslist_path, 'r').read()
  kernelslists = kernelslist.split('\n\n')

  xla_hlo_path_str = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  offline_execution_paths = list(xla_hlo_path.glob("./offline_execution_result*.csv"))
  offline_execution_objects = []
  for offline_execution_path in offline_execution_paths:
    print(offline_execution_path)
    offline_execution_object = csv.reader(open(offline_execution_path, 'r'))
    
    output_file = str(offline_execution_path).rsplit("/", 1)[1]
    output_path = f'/home/jueonpark/tracegen/offline_execution_result/{args.model}_{output_file}'
    output_file = open(output_path, 'w+')
    
    new_header = "GpuKernel,SimulationCycle\n"
    output_file.write(new_header)
    estimation_result = []
    estimation_result.append(new_header)
    prev_header = next(offline_execution_object)
    
    total_results = total_result.split('\n')
    for off_row in offline_execution_object:
      # find kernel number and real cycle
      found = False
      kernel_name = ""
      kernel_cycle = 0
      for kernel in kernelslists:
        # // Thunk: custom-call.84:__cudnn$convForward
        # // Kernel Name: _ZN5cudnn4gemm20computeOffsetsKernelENS0_20ComputeOffsetsParamsE
        # # NO_CXL
        # kernel-2627.traceg
        knl_elements = kernel.split('\n')
        if knl_elements[0].find(off_row[0]) != -1:
          found = True
          kernel_name = knl_elements[1].split(": ")[1]
          break
      if found:
        # find from total_results
        for tr_row in total_results:
          if tr_row.find(kernel_name) != -1:
            kernel_cycle = tr_row.split(',')[5]
            break
      # off_row[1] + ',' + \
      output_line = off_row[0] + ',' + str(kernel_cycle) + "\n"
      output_file.write(output_line)