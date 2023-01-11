import re
import os
import sys
from tqdm import tqdm
import shutil
import argparse
import pathlib

from hloutil import parse_thunk_schedule
from hloutil import parse_stats
from hloutil import HloDepdendencyManager

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="Model name", required=True)
parser.add_argument('-E', '--end', type=int, help="kernel end number", default=-1)

CUDA_SYNC = 'cudaDeviceSynchronize 0'

def instr_name(thunk):
  return thunk.split(':')[0]

def get_overlap_instructions(thunk):
  if thunk.find("Ndp") == -1:
    print("NOT A NDPX THUNK")
    exit(-1)
  overlap_instrs = thunk.split('$')[1].split(",")
  return overlap_instrs

def get_first_kernel_id(path):
  kernelslist_file = open(path)
  for line in kernelslist_file.readlines():
    if not 'kernel' in line:
      continue
    start = line.find('-') + 1
    end = line.find('.')
    return int(line[start:end])

# def construct_ndpx_sched_table(ndpx_sched_table_paths):
#   total_sched_table = {}
#   for ndpx_sched_table_path in ndpx_sched_table_paths:
#     ndpx_sched_table = open(ndpx_sched_table_path).read().split("\n")[1:-1]
#     for line in ndpx_sched_table:
#       ndpx_sched_info = line.split(",")
#       if ndpx_sched_info[0] in total_sched_table:
#         if ndpx_sched_info[8] in total_sched_table[ndpx_sched_info[0]]:
#           continue
#         else:
#           total_sched_table[ndpx_sched_info[0]].append(ndpx_sched_info[8])
#       else:
#         total_sched_table[ndpx_sched_info[0]] = []
#         total_sched_table[ndpx_sched_info[0]].append(ndpx_sched_info[8])
#   print(total_sched_table)
#   return total_sched_table

# find GPU HloInstructions to rewrite (writing to NDPX)]
def find_instrs_to_rewrite(hlo_graph_path, ndp_thunk_list):
  hlo_dependency_manager = HloDepdendencyManager(open(hlo_graph_path).read())
  ndp_instrs = [instr_name(ndp_thunk) for ndp_thunk in ndp_thunk_list]
  # simple. for all the NDPX co
  rewrite_instr_list = []
  for ndp_instr in ndp_instrs:
    parents = hlo_dependency_manager.hlo_table[ndp_instr]
    for parent in parents:
      if "get-tuple-element" in parent:
        parents += hlo_dependency_manager.hlo_table[parent]
    rewrite_instr_list += parents
  return rewrite_instr_list

# rewrite the GPU kernel file's STG address from 0x7* to 0x1007*
def rewrite_addr(original_filepath, new_filepath):
  print(f"rewrite to {original_filepath} -> {new_filepath}")
  file = open(original_filepath, "r")
  new_file = open(new_filepath, "w+")
  for line in tqdm(file.readlines()):
    if "STG" in line:
      for word in line.split():
        if "0x" in word:
          line = re.sub("0x7", "0x1007", line)
    new_file.write(line)
  file.close()
  new_file.close()
  return 

if __name__ == "__main__":
  args = parser.parse_args()
  # trace-related paths
  trace_path=f'/home/jueonpark/tracegen/traces/{args.model}/traces'
  stats_path=f'/home/jueonpark/tracegen/traces/{args.model}/traces/stats.csv'
  list_path=f'/home/jueonpark/tracegen/traces/{args.model}/traces/kernelslist'
  # xla_hlo-related paths
  xla_hlo_path_str=f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  graph_paths = list(xla_hlo_path.glob("*after_optimizations.txt"))
  # ndpx_sched_table_paths = list(xla_hlo_path.glob("ndpx_scheduling_table*"))
  ts_paths = list(xla_hlo_path.glob("*thunk_schedule"))
  ts_paths.sort(key=lambda x: str(x))
  # output trace dir path which all the traceg files woudld gather 
  # to generate a simulation environment.
  output_trace_dir=f'./traces/{args.model}/exp_trace_dir'

  # parse stats.csv
  stats_parsed = parse_stats(open(stats_path).read(), get_first_kernel_id(list_path), args.end)
  # for stats in stats_parsed:
  print("########## stats.csv parsed ##########")
  stats_idx = 0
  for ts_path in ts_paths:
    # for each cluster, parse thunk_schedule
    # ASSUMPTION: we assume that the thunk_schedule is sorted in timewise order
    print(f"current ts_path: {ts_path}")
    GPU_thunk_list, NDP_thunk_list = parse_thunk_schedule(open(ts_path).read())
    print("GPU_thunk_list:")
    print(GPU_thunk_list)

    ts_str = (ts_path.as_posix()).split("xla_hlo/")[1]
    print(ts_str)
    current_cluster = ts_str.split("_", 1)[1].split(".module")[0]
    print(f"current_cluster: {current_cluster}")
    current_module = ts_str.split(current_cluster + ".")[1].split(".thunk_schedule")[0]
    print(f"current_module: {current_module}")
    current_graph = ""
    for graph in graph_paths:
      if current_module in graph.as_posix():
        current_graph = graph
        break
    # list-up ndpx_traces
    ndpx_trace_dir_path=f'./traces/{args.model}/xla_hlo/packet_32_buffer_1_gpu_1_sync_0_simd_8/{current_cluster}'
    # ndpx_sched_table = construct_ndpx_sched_table(ndpx_sched_table_paths)
    ndpx_trace_files = os.listdir(ndpx_trace_dir_path)
    for i in range(10): # 왜 한 번에 안없어지지?
      for trace_file in ndpx_trace_files:
        if 'page_table' in trace_file:
          ndpx_trace_files.remove(trace_file)
        elif "_ON_THE_FLY_" in trace_file:
          ndpx_trace_files.remove(trace_file)
        elif "NdpEwiseFusedSeq" in trace_file:
          ndpx_trace_files.remove(trace_file)
        elif "noopt" in trace_file:
          ndpx_trace_files.remove(trace_file)
        elif "numbering" in trace_file:
          ndpx_trace_files.remove(trace_file)
        elif "memory" in trace_file:
          ndpx_trace_files.remove(trace_file)
    print(f"ndpx_trace_files: {ndpx_trace_files}")
    # 어차피 지금은 on-the-fly가 없으니 상관이 없다

    instrs_to_rewrite = find_instrs_to_rewrite(os.path.join(xla_hlo_path_str, current_graph),\
                                               NDP_thunk_list)

    # now, match the GPU thunk list and stats_parsed
    gpu_match_table = {}
    gpu_thunk_idx = 0
    gpu_match_table["unmatched"] = []
    while gpu_thunk_idx <= len(GPU_thunk_list) - 1:
      print(f"gpu_thunk: {GPU_thunk_list[gpu_thunk_idx]}, stats: {stats_parsed[stats_idx][0]}")
      while stats_parsed[stats_idx][0].find("Eigen") != -1:
        if stats_idx < len(stats_parsed) - 1:
          stats_idx += 1

      if GPU_thunk_list[gpu_thunk_idx].find("custom-call") != -1:
        if stats_parsed[stats_idx][0].find("gemm") != -1:
          gpu_match_table[instr_name(GPU_thunk_list[gpu_thunk_idx])] = [stats_parsed[stats_idx]]
          gpu_thunk_idx += 1
          stats_idx += 1
          continue
        elif stats_parsed[stats_idx][0].find("conv") != -1:
          gpu_match_table[instr_name(GPU_thunk_list[gpu_thunk_idx])] = [stats_parsed[stats_idx]]
          gpu_thunk_idx += 1
          stats_idx += 1
          continue
        elif stats_parsed[stats_idx][0].find("scudnn") != -1:
          gpu_match_table[instr_name(GPU_thunk_list[gpu_thunk_idx])] = [stats_parsed[stats_idx]]
          gpu_thunk_idx += 1
          stats_idx += 1
          continue
        else:
          print(stats_parsed[stats_idx])
          gpu_match_table["unmatched"].append(stats_parsed[stats_idx])
          stats_idx += 1
          continue
      
      elif (stats_parsed[stats_idx][0].find("_") == -1) and \
            stats_parsed[stats_idx][0] == GPU_thunk_list[gpu_thunk_idx]:
        # case for things such as "fusion" "add" without *.{num}
        gpu_match_table[GPU_thunk_list[gpu_thunk_idx]] = [stats_parsed[stats_idx]]
        stats_idx += 1
        gpu_thunk_idx += 1
        continue

      elif stats_parsed[stats_idx][0].replace("__", ".").replace("_", ".")\
              .find(GPU_thunk_list[gpu_thunk_idx]) != -1:
        gpu_match_table[GPU_thunk_list[gpu_thunk_idx]] = []
        while stats_parsed[stats_idx][0].replace("__", ".").replace("_", ".").find(GPU_thunk_list[gpu_thunk_idx]) != -1:
          gpu_match_table[GPU_thunk_list[gpu_thunk_idx]].append(stats_parsed[stats_idx])
          if stats_idx < len(stats_parsed) - 1:
            stats_idx += 1
          else:
            break
        gpu_thunk_idx += 1
        continue
      
      else:
        print("unmatched:")
        print(f"gpu_thunk: {GPU_thunk_list[gpu_thunk_idx]}, stats: {stats_parsed[stats_idx][0]}")
        gpu_match_table["unmatched"].append(stats_parsed[stats_idx])
        stats_idx += 1
      # end of GPU thunk - stats.csv matching for current cluster

    #################### start making NDP experiment directories ####################
    print("NDP_thunk_list:")
    print(NDP_thunk_list)

    for ndp_thunk in tqdm(NDP_thunk_list):
      print(f"ndp_thunk: {ndp_thunk}")
      ndp_call = instr_name(ndp_thunk)
      new_ndp_path = f"{current_cluster}_{ndp_call}"
      print(f"new_ndp_path: {new_ndp_path}")
      # find corresponding file
      for file in ndpx_trace_files:
        if file.find(ndp_call) != -1:
          # found the file!
          new_ndpx_dir_path = os.path.join(output_trace_dir, new_ndp_path)
          new_ndpx_gpu0_path = os.path.join(new_ndpx_dir_path, "GPU_0")
          os.makedirs(new_ndpx_gpu0_path, exist_ok=True)

          ndpx_file_path = os.path.join(ndpx_trace_dir_path, file)
          new_ndpx_file_path = os.path.join(new_ndpx_gpu0_path, file)
          shutil.copy(ndpx_file_path, new_ndpx_file_path)
      
          # move overlapping GPU kernels to NDPX directory
          overlap_instrs = get_overlap_instructions(ndp_thunk)
          gpu_traces = []
          for overlap_instr in overlap_instrs:
            print(f"overlap_instr: {overlap_instr}")
            kernel_pairs = []
            try:
              kernel_pairs = gpu_match_table[overlap_instr]
            except:
              print("WRONG SCHEDULING")
              pass
              # exit(-1)
            for kernel_pair in kernel_pairs:
              print(f"kernel_name: {kernel_pair[0]}, kernel_num: {kernel_pair[1]}")
              kernel_file = f'kernel-{kernel_pair[1]}.traceg'
              gpu_traces.append(kernel_file)
              kernel_file_path = os.path.join(trace_path, kernel_file)
              new_kernel_file_path = os.path.join(new_ndpx_gpu0_path, kernel_file)
              if overlap_instr in instrs_to_rewrite:
                rewrite_addr(kernel_file_path, new_kernel_file_path)
              else:
                shutil.copy(kernel_file_path, new_kernel_file_path)
              try:
                gpu_match_table.pop(overlap_instr)
              except:
                pass

          # write base kernelslist.g file.
          kernelslist_base_file_path = os.path.join(new_ndpx_dir_path, "kernelslist.g")
          kernelslist_base_file = open(kernelslist_base_file_path, "w+")
          for gpu_trace in gpu_traces:
            kernelslist_base_file.write(gpu_trace + '\n')
          kernelslist_base_file.write(CUDA_SYNC)
          kernelslist_base_file.close()

          # write kernelslist.g file in GPU_0.
          kernelslist_file_path = os.path.join(new_ndpx_gpu0_path, "kernelslist.g")
          kernelslist_file = open(kernelslist_file_path, "w+")
          kernelslist_file.write(file + '\n')
          for gpu_trace in gpu_traces:
            kernelslist_file.write(gpu_trace + '\n')
          kernelslist_file.close()
          break

    #################### start making GPU experiment directories ####################
    for gpu_instr in tqdm(gpu_match_table):
      for gpu_pair in gpu_match_table[gpu_instr]:
        kernel_name = gpu_pair[0]
        kernel_num = gpu_pair[1]
        print(f"handling {kernel_num}, {kernel_name}")
        kernel_file = f'kernel-{kernel_num}.traceg'
        kernel_file_path = os.path.join(trace_path, kernel_file)
        new_kernel_dir_path = os.path.join(output_trace_dir, str(kernel_num))
        new_kernel_gpu0_path = os.path.join(new_kernel_dir_path, "GPU_0")
        os.makedirs(new_kernel_gpu0_path, exist_ok=True)
        new_kernel_file_path = os.path.join(new_kernel_gpu0_path, kernel_file)
        if gpu_instr in instrs_to_rewrite:
          rewrite_addr(kernel_file_path, new_kernel_file_path)
        else:
          shutil.copy(kernel_file_path, new_kernel_file_path)
        # write kernelslist.g file.
        kernelslist_base_file_path = os.path.join(new_kernel_dir_path, "kernelslist.g")
        kernelslist_base_file = open(kernelslist_base_file_path, "w+")
        kernelslist_base_file.write(kernel_file + '\n')
        kernelslist_base_file.write(CUDA_SYNC)
        kernelslist_base_file.close()
        # write kernelslist.g file in GPU_0.
        kernelslist_file_path = os.path.join(new_kernel_gpu0_path, "kernelslist.g")
        kernelslist_file = open(kernelslist_file_path, "w+")
        kernelslist_file.write(kernel_file + '\n')
        kernelslist_file.close()