# merges offline_execution_result.csv and simulation results

# input: gpu_kernel_estimation_table_cluster_0.csv
# merges estimated gpu cycles and real gpu cycles
import csv
import argparse
import pathlib
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="model", required=True)

if __name__ == "__main__":
  args = parser.parse_args()
  
  # simulation result file
  sim_result_file = f'/home/jueonpark/tracegen/csv_files/{args.model}-NDPX_baseline_64-1-nosync.csv'
  sim_results = open(sim_result_file, 'r').read().split("\n")

  # finding offline_execution_result*.csv files
  xla_hlo_path_str = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  offline_execution_paths = list(xla_hlo_path.glob("./offline_execution_result*.csv"))
  offline_execution_objects = []

  sim_result_idx = 1
  # ASSUMPTION: the offline_execution_paths are well-sorted.
  for offline_execution_path in offline_execution_paths:
    print(offline_execution_path)
    offline_execution_object = csv.reader(open(offline_execution_path, 'r'))
    # output file to write ({model_name}_{offline_execution_result*.csv})
    output_file = str(offline_execution_path).rsplit("/", 1)[1]
    output_path = f'/home/jueonpark/tracegen/offline_execution_result/{args.model}_{output_file}'
    output_file = open(output_path, 'w+')    
    new_header = "hlo_op,runtime\n"
    output_file.write(new_header)
    
    next(offline_execution_object)
    for off_row in offline_execution_object:
      # find kernel number and real cycle
      kernel_cycle = 0
      sim_result = sim_results[sim_result_idx].split(",")
      print(f"current: {off_row[0]}, {sim_result[4]}")

      if off_row[0].find("copy") != -1:
        # copy is not generated to thunk.
        output_line = off_row[0] + ',' + "-1\n"
        output_file.write(output_line)

      elif off_row[0].find("custom-call") != -1:
        # custom-call
        if sim_result[4].find("cu") != -1:
          # found?
          kernel_cycle = sim_result[5]
          output_line = off_row[0] + ',' + str(kernel_cycle) + "\n"
          output_file.write(output_line)
          sim_result_idx += 1  
        elif sim_result[4].find("gemm") != -1:
          kernel_cycle = sim_result[5]
          output_line = off_row[0] + ',' + str(kernel_cycle) + "\n"
          output_file.write(output_line)
          sim_result_idx += 1
      
      elif sim_result[4].replace("__", ".").replace("_", ".").find(off_row[0]) != -1:
        # XLA-lowered operation found!
        while sim_results[sim_result_idx].split(",")[4].replace("__", ".").replace("_", ".").find(off_row[0]) != -1:
          kernel_cycle += int(sim_result[5])
          sim_result_idx += 1
        output_line = off_row[0] + ',' + str(kernel_cycle) + "\n"
        output_file.write(output_line)