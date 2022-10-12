import os
import sys
import shutil
import argparse
import pathlib

from hloutil import parse_thunk_schedule
from hloutil import parse_stats

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="Model name", required=True)
parser.add_argument('-E', '--end', type=int, help="kernel end number", default=-1)

CUDA_SYNC = 'cudaDeviceSynchronize 0'

def match(ts_parsed, stats_parsed):
  ts_matched = []
  stats_matched = []
  ts_unmatched = [i for i in ts_parsed]
  stats_unmatched = [i for i in stats_parsed]

  for order, thunk_name in ts_parsed:
    stats_matched_chunk = []
    unmatched_cpy = stats_unmatched.copy()
    for kernel_no, kernel_name in unmatched_cpy:
      # for Transformer and BERT
      if 'custom-call' in thunk_name and 'gemm' in kernel_name:
        ts_unmatched.remove((order, thunk_name))
        stats_unmatched.remove((kernel_no, kernel_name))
        ts_matched.append((order,thunk_name))
        stats_matched.append([(kernel_no, kernel_name)])
        break
      # for CNNs
      elif 'custom-call' in thunk_name and 'conv' in kernel_name:
        ts_unmatched.remove((order, thunk_name))
        stats_unmatched.remove((kernel_no, kernel_name))
        ts_matched.append((order,thunk_name))
        stats_matched.append([(kernel_no, kernel_name)])
        break
      # for XLA-generated kernels
      elif thunk_name.replace(".", "_").replace("-", "_") == kernel_name:
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
        stats_unmatched.remove((kernel_no, kernel_name))
        stats_matched_chunk.append((kernel_no, kernel_name))
    if len(stats_matched_chunk) > 0:
      stats_matched.append(stats_matched_chunk)

  return (ts_matched, stats_matched), (ts_unmatched, stats_unmatched)

def gpu_instr_name(thunk):
  return thunk.split(':')[0]

def get_first_kernel_id(path):
  kernelslist_file = open(path)
  for line in kernelslist_file.readlines():
    if not 'kernel' in line:
      continue
    start = line.find('-') + 1
    end = line.find('.')
    return int(line[start:end])

def count_comma(txt):
  result = 0
  for char in txt:
    if char == ',':
      result += 1
  return result

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
  ts_paths = list(xla_hlo_path.glob("*thunk_schedule"))
  ndpx_trace_dir_path=f'./traces/{args.model}/xla_hlo/packet_32_buffer_1_gpu_1_sync_0_simd_8'  # for NdpEwiseFused files
  # output trace dir path which all the traceg files woudld gather to generate a simulation environment.
  output_trace_dir=f'./traces/{args.model}/exp_trace_dir'

  # parsed thunk schedule and stats.csv
  GPU_thunks, NDP_thunks = [], []
  for ts_path in ts_paths:
    GPU_thunk_list, NDP_thunk_list = parse_thunk_schedule(open(ts_path).read())
    GPU_thunks.extend(GPU_thunk_list)
    NDP_thunks.extend(NDP_thunk_list)
  stats_parsed = parse_stats(open(stats_path).read(), get_first_kernel_id(list_path), args.end)
  (GPU_thunks_matched, stats_matched), (GPU_thunks_unmatched, stats_unmatched) \
      = match(GPU_thunks, stats_parsed)
  matched = list(zip(GPU_thunks_matched, stats_matched))

  # list-up ndpx_traces
  ndpx_trace_files = os.listdir(ndpx_trace_dir_path)
  for i in range(3):
    for trace_file in ndpx_trace_files:
      if 'page_table' in trace_file:
        ndpx_trace_files.remove(trace_file)
      elif "_ON_THE_FLY_" in trace_file:
        ndpx_trace_files.remove(trace_file)
  
  # make directories for each NDPX kernel
  for file in ndpx_trace_files:
    print(file)
    # make directory for each NDPX trace file
    ndpx_name = file.split("_0_")[0]  # this would be the name of the NDPX directory
    new_ndpx_dir_path = os.path.join(output_trace_dir, ndpx_name)
    new_ndpx_gpu0_path = os.path.join(new_ndpx_dir_path, "GPU_0")
    os.makedirs(new_ndpx_gpu0_path, exist_ok=True)
    ndpx_file_path = os.path.join(ndpx_trace_dir_path, file)
    new_ndpx_file_path = os.path.join(new_ndpx_gpu0_path, file)
    shutil.move(ndpx_file_path, new_ndpx_file_path)
    # find GPU kernels that is to be scheduled to NDPX kernel
    overlap_candidate_str = (file.split("$")[1]).split(".traceg")[0]
    overlap_candidates = []
    if count_comma(overlap_candidate_str) > 0:
      # multiple overlapping
      overlap_candidates_ = overlap_candidate_str.split(",")
      overlap_candidates.extend(overlap_candidates_)
    else:
      # one-by-one overlapping
      overlap_candidates.append(overlap_candidate_str)
    # find corresponding GPU kernels and move the kernel files to ndpx_dir_path
    gpu_traces = []
    for (order, thunk_name), kernels in matched:
      gpu_instr = gpu_instr_name(thunk_name)
      if gpu_instr in overlap_candidates:
        print(gpu_instr)
        print(kernels)
        for kernel_no, kernel_name in kernels:
          kernel_file = f'kernel-{kernel_no}.traceg'
          gpu_traces.append(kernel_file)
          kernel_file_path = os.path.join(trace_path, kernel_file)
          new_kernel_file_path = os.path.join(new_ndpx_gpu0_path, kernel_file)
          shutil.move(kernel_file_path, new_kernel_file_path)
        matched.remove(((order, thunk_name), kernels))
    # write kernelslist.g file.
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
    
  # make directories for each GPU kernel
  for (_, thunk_name), kernels in matched:
    for kernel_no, kernel_name in kernels:
      kernel_file = f'kernel-{kernel_no}.traceg'
      kernel_file_path = os.path.join(trace_path, kernel_file)
      new_kernel_dir_path = os.path.join(output_trace_dir, str(kernel_no))
      new_kernel_gpu0_path = os.path.join(new_kernel_dir_path, "GPU_0")
      os.makedirs(new_kernel_gpu0_path, exist_ok=True)
      new_kernel_file_path = os.path.join(new_kernel_gpu0_path, kernel_file)
      shutil.move(kernel_file_path, new_kernel_file_path)
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