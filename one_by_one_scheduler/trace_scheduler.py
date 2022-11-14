import os 
import argparse
import pathlib

from hloutil import parse_thunk_schedule
from hloutil import parse_stats
from hloutil import HloDepdendencyManager

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="Model name", required=True)
parser.add_argument('-E', '--end', type=int, help="kernel end number", default=-1)

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
        print("39")
        print(thunk_name)
        print(kernel_name)
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
        print("51")
        print(thunk_name)
        print(kernel_name)
        stats_unmatched.remove((kernel_no, kernel_name))
        stats_matched_chunk.append((kernel_no, kernel_name))
    if len(stats_matched_chunk) > 0:
      stats_matched.append(stats_matched_chunk)

  return (ts_matched, stats_matched), (ts_unmatched, stats_unmatched)

def custom_call_name(thunk):
  return thunk.split(':')[0]

def list_substr_contains(list_, str_):
  result = False
  for entry in list_:
    result = result or str_ in entry
  return result

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
  ndpx_trace_dir_path=f'./traces/{args.model}/xla_hlo/packet_32_buffer_1_gpu_1_sync_0_simd_8'  # for NdpEwiseFused files
  output=f'/home/jueonpark/tracegen/traces/{args.model}/kernelslist.g'

  # thunk lists & hlo map
  GPU_thunks, NDP_thunks = [], []
  for ts_path in ts_paths:
    GPU_thunk_list, NDP_thunk_list = parse_thunk_schedule(open(ts_path).read())
    GPU_thunks.extend(GPU_thunk_list)
    NDP_thunks.extend(NDP_thunk_list)
  # parsed stats.csv
  stats_parsed = parse_stats(open(stats_path).read(), get_first_kernel_id(list_path), args.end)
  # matching parsed stats.csv and thunks
  (GPU_thunks_matched, stats_matched), (GPU_thunks_unmatched, stats_unmatched) \
      = match(GPU_thunks, stats_parsed)
  matched = list(zip(GPU_thunks_matched, stats_matched))

  finish_list = []
  num_gpu_custom_call = 0
  scheduled_kernels = dict()
  scheduled_kernels[''] = []
  no_cxl_flags = dict()
  for order, thunk in GPU_thunks:
    name = custom_call_name(thunk)
    scheduled_kernels[name] = []
    no_cxl_flags[name] = True
    if 'custom-call' in thunk:
      num_gpu_custom_call +=1

  overlapped_candidates = []
  
  # ndpx_traces for scheduling cost-model compiler results
  ndpx_trace_files = os.listdir(ndpx_trace_dir_path)
  for i in range(3):
    for trace_file in ndpx_trace_files:
      if 'page_table' in trace_file:
        ndpx_trace_files.remove(trace_file)
  print("list of ndpx trace files to schedule")
  for file in ndpx_trace_files:
    print(file)

  ndpx_trace_files_to_write = ndpx_trace_files.copy()

  """
  fusion offloading overlapping
  warning: there might be ndp kernels that do not have overlapping kernel
  NDPX trace format: _NDP_custom-call.149_0_NdpEwiseFused$fusion.14.traceg
  """
  NDP_thunks_cpy = NDP_thunks.copy()
  for order, ndp_thunk_name in NDP_thunks_cpy:
    ndp_custom_call = custom_call_name(ndp_thunk_name)
    if "NdpEwiseFused" in ndp_thunk_name:
      print(f"ndp_thunk_name: {ndp_thunk_name}")
      # the custom call target itself do not have overlapping information, so find the 
      # overlapping target from the file.
      # file format:
      # _NDP_custom-call.200_0_NdpEwiseFused$fusion.438.traceg
      # _ON_THE_FLY_custom-call.133_0_NdpEwiseFusedOnTheFly$fusion.201.traceg
      # _ON_THE_FLY_custom-call.133_2_NdpEwiseFusedOnTheFly$fusion.197.traceg
      for trace_file in ndpx_trace_files:
        if ndp_custom_call in trace_file:
          # the trace file have the overlapping information 
          overlap_candidate_name = (trace_file.split("$")[1]).split(".traceg")[0]
          # try overlapping for identified gpu thunk
          for gpu_order, gpu_thunk_name in GPU_thunks:
            gpu_custom_call = custom_call_name(gpu_thunk_name)
            if gpu_custom_call == overlap_candidate_name:
              # there is overlap candidate in GPU kernel (typical case)
              overlap_exist = True
              overlapped_candidates.append(gpu_custom_call)
              scheduled_kernels[gpu_custom_call].append(trace_file)
              print(f'NDP({trace_file}) mapped to {gpu_custom_call}')
              if "NdpEwiseFusedOnTheFly" in ndp_thunk_name:
                no_cxl_flags[gpu_custom_call] = False
              ndpx_trace_files.remove(trace_file)
              break

  # file to write:
  # - kernelslsit.g in the model directory
  f = open(output, "w+")
  for (order, thunk_name), kernels in matched:
    gpu_custom_call = custom_call_name(thunk_name)
    for kernel_no, kernel_name in kernels:
      f.write(f'// Thunk: {thunk_name}\n')
      f.write(f'// Kernel Name: {kernel_name}\n')
      if no_cxl_flags[gpu_custom_call]:
        f.write(f'# NO_CXL\n')
      for ndp_thunk_name in scheduled_kernels[gpu_custom_call]:
        if 'NdpEwiseFused' in ndp_thunk_name:
          ndp_custom_call = custom_call_name(ndp_thunk_name)
          # make on-the-fly trace files to be written.
          # TODO: multiple scheduling + _ON_THE_FLY_ handling
          for trace_file in ndpx_trace_files_to_write:
            if ('NdpEwiseFused' in trace_file) and (ndp_custom_call in trace_file):
              f.write(f'{trace_file}\n')
              # if '_ON_THE_FLY' in trace_file:
              #   f.write(f'_BAR_\n')
        else:
          f.write(f'{ndp_thunk_name}\n')
      f.write(f'kernel-{kernel_no}.traceg\n')
      f.write('\n')
    if 'custom-call' in thunk_name:
      num_gpu_custom_call -= 1
    finish_list.append(thunk_name.split(':')[0])
  first = True
  for kernel_no, kernel_name in stats_unmatched:
    f.write(f'// Kernel Name: {kernel_name}\n')
    f.write(f'# NO_CXL\n')
    if first:
      first = False
      for remains in scheduled_kernels['']:
        f.write(remains)
        f.write('\n')
    f.write(f'kernel-{kernel_no}.traceg\n')
    f.write('\n')