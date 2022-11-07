# merges offline_execution_result.csv and simulation results

# input: gpu_kernel_estimation_table_cluster_0.csv
# merges estimated gpu cycles and real gpu cycles
import re
import csv
import argparse
import pathlib

from hloutil import parse_thunk_schedule
from hloutil import parse_stats

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="model", required=True)

def match(ts_parsed, stats_parsed):
  ts_matched = []
  stats_matched = []
  ts_unmatched = [i for i in ts_parsed]
  stats_unmatched = [i for i in stats_parsed]

  stats_unmatched_cpy = stats_unmatched.copy()
  for kernel_no, kernel_name in stats_unmatched_cpy:
    print(f"kernel_no: {kernel_no}, kernel_name: {kernel_name}")
    stats_matched_chunk = []
    for order, thunk_name in ts_parsed:
      # for Transformer and BERT
      if 'custom-call' in thunk_name and 'gemm' in kernel_name:
        # print(f"kernel_no: {kernel_no}, kernel_name: {kernel_name}")
        ts_unmatched.remove((order, thunk_name))
        stats_unmatched.remove((kernel_no, kernel_name))
        ts_matched.append((order,thunk_name))
        stats_matched.append([(kernel_no, kernel_name)])
        break
      # for CNNs
      elif 'custom-call' in thunk_name and 'conv' in kernel_name:
        # print(f"kernel_no: {kernel_no}, kernel_name: {kernel_name}")
        ts_unmatched.remove((order, thunk_name))
        stats_unmatched.remove((kernel_no, kernel_name))
        ts_matched.append((order,thunk_name))
        stats_matched.append([(kernel_no, kernel_name)])
        break
      elif 'custom-call' in thunk_name and 'cu' in kernel_name:
        # print(f"kernel_no: {kernel_no}, kernel_name: {kernel_name}")
        ts_unmatched.remove((order, thunk_name))
        stats_unmatched.remove((kernel_no, kernel_name))
        ts_matched.append((order,thunk_name))
        stats_matched.append([(kernel_no, kernel_name)])
        break
      # for XLA-generated kernels
      elif thunk_name.replace(".", "_").replace("-", "_") == kernel_name:
        print(f"thunk_name: {thunk_name}, kernel_name: {kernel_name}")
        try:
          ts_unmatched.remove((order, thunk_name))
          ts_matched.append((order, thunk_name))
          stats_matched_chunk.append((kernel_no, kernel_name))
          stats_unmatched.remove((kernel_no, kernel_name))
        except:
          print("qwer")
          continue
      # for XLA-generated kernels
      elif '__' in kernel_name and thunk_name.replace(".", "_").replace("-", "_") == kernel_name[:kernel_name.find('__')]:
        # print(f"thunk_name: {thunk_name}, kernel_name: {kernel_name}")
        stats_unmatched.remove((kernel_no, kernel_name))
        stats_matched_chunk.append((kernel_no, kernel_name))
    if len(stats_matched_chunk) > 0:
      stats_matched.append(stats_matched_chunk)

  return (ts_matched, stats_matched), (ts_unmatched, stats_unmatched)

def get_first_kernel_id(path):
  kernelslist_file = open(path)
  for line in kernelslist_file.readlines():
    if not 'kernel' in line:
      continue
    start = line.find('-') + 1
    end = line.find('.')
    return int(line[start:end])

if __name__ == "__main__":
  args = parser.parse_args()
  # trace-related paths
  stats_path=f'/home/jueonpark/tracegen/traces/{args.model}/traces/stats.csv'
  list_path=f'/home/jueonpark/tracegen/traces/{args.model}/traces/kernelslist'
  # xla_hlo-related paths
  xla_hlo_path_str=f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  graph_paths = list(xla_hlo_path.glob("*after_optimizations.txt"))
  ts_paths = list(xla_hlo_path.glob("*thunk_schedule"))
  # simulation result
  csv_path = f'/home/jueonpark/tracegen/csv_files/{args.model}-NDPX_baseline_64-1-nosync.csv'

  # thunk lists & hlo map
  GPU_thunks, NDP_thunks = [], []
  for ts_path in ts_paths:
    GPU_thunk_list, NDP_thunk_list = parse_thunk_schedule(open(ts_path).read())
    GPU_thunks.extend(GPU_thunk_list)
    NDP_thunks.extend(NDP_thunk_list)
  # parsed stats.csv
  stats_parsed = parse_stats(open(stats_path).read(), get_first_kernel_id(list_path), -1)
  # matching parsed stats.csv and thunks
  (GPU_thunks_matched, stats_matched), (GPU_thunks_unmatched, stats_unmatched) \
      = match(GPU_thunks, stats_parsed)
  matched = list(zip(GPU_thunks_matched, stats_matched))

  total_result = open(csv_path, 'r').read()

  xla_hlo_path_str = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  offline_execution_posix_paths = list(xla_hlo_path.glob("./offline_execution_result*.csv"))
  offline_execution_paths = []
  for posix_path in offline_execution_posix_paths:
    if str(posix_path).find("cluster") != -1:
      offline_execution_paths.append(str(posix_path))
  # sort the offline execution paths based on "cluster" prefix
  offline_execution_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
  for offline_execution_path in offline_execution_paths:
    print(offline_execution_path)
    offline_execution_object = csv.reader(open(offline_execution_path, 'r'))
    output_file = str(offline_execution_path).rsplit("/", 1)[1]
    output_path = f'/home/jueonpark/tracegen/offline_execution_result/{args.model}_{output_file}'
    output_file = open(output_path, 'w+')
    
    new_header = "hlo_op,runtime\n"
    output_file.write(new_header)
    estimation_result = []
    estimation_result.append(new_header)
    prev_header = next(offline_execution_object)
    total_results = total_result.split('\n')
    i = 0
    for off_row in offline_execution_object:
      print(f"{i}: {off_row}")
      i += 1
      # find kernel number and real cycle
      found = False
      kernel_name = ""
      kernel_cycle = 0
      for ts_pair, trace_pair in matched:
        if off_row[0] == ts_pair[1]:
          print(f"{ts_pair}, {trace_pair}")
          found = True
          kernel_name = trace_pair[0][1]
          # matched.remove((ts_pair, trace_pair))
          # find the 
          break
      if found:
        # find from total_results
        for tr_row in total_results:
          if tr_row.find(kernel_name) != -1:
            kernel_cycle = tr_row.split(',')[5]
            break
      output_line = off_row[0] + ',' + str(kernel_cycle) + "\n"
      output_file.write(output_line)