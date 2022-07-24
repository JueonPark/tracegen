from inspect import trace
import os 
import argparse

from utils2 import parse_thunk_schedule
from utils2 import parse_stats
from utils2 import HloDepdendencyManager

def match(ts_parsed, stats_parsed):
  ts_matched = []
  stats_matched = []
  ts_unmatched = [i for i in ts_parsed]
  stats_unmatched = [i for i in stats_parsed]

  for order, thunk_name in ts_parsed:
    stats_matched_chunk = []
    unmatched_cpy = stats_unmatched.copy()
    for kernel_no, kernel_name in unmatched_cpy:
      if 'custom-call' in thunk_name and 'gemm' in kernel_name:
        ts_unmatched.remove((order, thunk_name))
        stats_unmatched.remove((kernel_no, kernel_name))
        ts_matched.append((order,thunk_name))
        stats_matched.append([(kernel_no, kernel_name)])
        break
      elif thunk_name.replace(".", "_").replace("-", "_") == kernel_name:
        ts_unmatched.remove((order,thunk_name))
        ts_matched.append((order,thunk_name))
        stats_matched_chunk.append((kernel_no, kernel_name))
        stats_unmatched.remove((kernel_no, kernel_name))
      elif '__' in kernel_name and thunk_name.replace(".", "_").replace("-", "_") == kernel_name[:kernel_name.find('__')]:
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

# fw-hops argument:
# - for xxx, fw-hops should be 29
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="Model name", required=True)
parser.add_argument('-E', '--end', type=int, help="kernel end number", default=-1)
# bert large, batch 2, one encoder layer: 13
# bert large, batch 3, three encoder layer: 31
parser.add_argument('-f', '--fw-hops', type=int, help="forward kernel hops", default=31)

args = parser.parse_args()

stats_path=f'/home/jueonpark/tracegen/traces/{args.model}/traces/stats.csv'
list_path=f'/home/jueonpark/tracegen/traces/{args.model}/traces/kernelslist'
ts_path=f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/module_0000.thunk_schedule'
# graph_path=f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/{args.model}.txt'
graph_path=f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/after_optimizations.mlir'
ndpx_trace_dir_path=f'./traces/{args.model}/xla_hlo/packet_32_buffer_1_gpu_1_sync_0_simd_8'  # for NdpEwiseFused files
output=f'/home/jueonpark/tracegen/traces/{args.model}/kernelslist.g'

GPU_thunks, NDP_thunks = parse_thunk_schedule(open(ts_path).read())
stats_parsed = parse_stats(open(stats_path).read(), get_first_kernel_id(list_path), args.end)
manager = HloDepdendencyManager(open(graph_path).read())
(GPU_thunks_matched, stats_matched), (GPU_thunks_unmatched, stats_unmatched) = match(GPU_thunks, stats_parsed)
matched = list(zip(GPU_thunks_matched, stats_matched))

hops_map = manager.get_hops_map()
for key in hops_map:
  if not 'custom-call' in hops_map:
    hops_map.remove(key)

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
print(ndpx_trace_files)
"""
Forward schedule
Step 1, Schedule Dgrad dependent NDP kernels (ex: LayerNorm)
"""
NDP_thunks_cpy = NDP_thunks.copy()
for order, ndp_thunk_name in NDP_thunks_cpy:
  ndp_custom_call = custom_call_name(ndp_thunk_name)
  ndp_hops = manager.get_custom_call_hops(ndp_custom_call)
  if 'SoftmaxForward' in ndp_thunk_name :
    overlap_gpu_thunk = ''
    ph1_used = False
    ph3_used = False
    for gpu_order, gpu_thunk_name in GPU_thunks:
      gpu_custom_call = custom_call_name(gpu_thunk_name)
      gpu_hops = manager.get_custom_call_hops(gpu_custom_call)
      if 'custom-call' in gpu_thunk_name and \
          manager.is_dependent(ndp_custom_call, gpu_custom_call) and \
          ndp_hops == gpu_hops + 1:
        if ph1_used:
          print("ERROR: this IS the phase 1")
          exit()
        no_cxl_flags[gpu_custom_call] = False
        scheduled_kernels[gpu_custom_call].append(f'_ON_THE_FLY_{ndp_custom_call}_fw_bert_softmax_ph1.traceg')
        ph1_used = True
      elif 'custom-call' in gpu_thunk_name and \
          not manager.is_dependent(ndp_custom_call, gpu_custom_call) and \
          ndp_hops == gpu_hops + 2:
        if ph3_used:
          print("ERROR: this IS the phase 3")
          exit()
        scheduled_kernels[gpu_custom_call].append(f'_NDP_{ndp_custom_call}_fw_bert_softmax_reduce_max_1.traceg')
        scheduled_kernels[gpu_custom_call].append(f'_BAR_')
        scheduled_kernels[gpu_custom_call].append(f'_NDP_{ndp_custom_call}_fw_bert_softmax_reduce_max_2.traceg')
        scheduled_kernels[gpu_custom_call].append(f'_BAR_')
        scheduled_kernels[gpu_custom_call].append(f'_NDP_{ndp_custom_call}_fw_bert_softmax_reduce_accum_1.traceg')
        scheduled_kernels[gpu_custom_call].append(f'_BAR_')
        scheduled_kernels[gpu_custom_call].append(f'_NDP_{ndp_custom_call}_fw_bert_softmax_reduce_accum_2.traceg')
        scheduled_kernels[gpu_custom_call].append(f'_BAR_')
        scheduled_kernels[gpu_custom_call].append(f'_NDP_{ndp_custom_call}_fw_bert_softmax_mul_and_dropout.traceg')  
        ph3_used = True
        hops_map[gpu_custom_call] = ndp_hops
    NDP_thunks.remove((order, ndp_thunk_name))
    
    
  elif 'ForwardBertOutput' in ndp_thunk_name:
    overlap_gpu_thunk = ''
    ph1_used = False
    ph3_used = False
    for gpu_order, gpu_thunk_name in GPU_thunks:
      gpu_custom_call = custom_call_name(gpu_thunk_name)
      gpu_hops = manager.get_custom_call_hops(gpu_custom_call)
      if 'custom-call' in gpu_thunk_name and \
          manager.is_dependent(ndp_custom_call, gpu_custom_call) and \
          ndp_hops == gpu_hops + 1:
        no_cxl_flags[gpu_custom_call] = False
        scheduled_kernels[gpu_custom_call].append(f'_ON_THE_FLY_{ndp_custom_call}_fw_bert_output_ph1.traceg')
        scheduled_kernels[gpu_custom_call].append(f'_BAR_')
        scheduled_kernels[gpu_custom_call].append(f'_NDP_{ndp_custom_call}_fw_bert_output_ph2_mean.traceg')
        scheduled_kernels[gpu_custom_call].append(f'_BAR_')
        scheduled_kernels[gpu_custom_call].append(f'_NDP_{ndp_custom_call}_fw_bert_output_ph2_var.traceg')
        print(f'NDP({ndp_thunk_name}) mapped to {gpu_thunk_name}')
        if args.fw_hops - ndp_hops <= 2:
          print(gpu_custom_call)
          scheduled_kernels[gpu_custom_call].append(f'_BAR_')
          scheduled_kernels[gpu_custom_call].append(f'_ON_THE_FLY_{ndp_custom_call}_fw_bert_output_ph3.traceg')
          print(f'NDP({ndp_thunk_name}) mapped to {gpu_thunk_name}')
          ph3_used = True
        if ph1_used:
          print("ERROR")
          exit()
        ph1_used = True
      elif manager.need_bert_ph3_scheduling(gpu_custom_call, ndp_custom_call) \
          and not ph3_used:
        scheduled_kernels[gpu_custom_call].append(f'_ON_THE_FLY_{ndp_custom_call}_fw_bert_output_ph3.traceg')
        ph3_used = True
        print(f'NDP({ndp_thunk_name}) mapped to {gpu_thunk_name}')
    NDP_thunks.remove((order, ndp_thunk_name))

"""
Backward schedule
Step 1, Schedule Dgrad dependent NDP kernels (ex: LayerNorm)
"""
NDP_thunks_cpy = NDP_thunks.copy()
for order, ndp_thunk_name in NDP_thunks_cpy:
  ndp_custom_call = custom_call_name(ndp_thunk_name)
  if 'BackwardBertOutput' in ndp_thunk_name or 'SoftmaxBackward' in ndp_thunk_name:
    overlap_gpu_thunk = ''
    for gpu_order, gpu_thunk_name in GPU_thunks:
      gpu_custom_call = custom_call_name(gpu_thunk_name)
      if 'custom-call' in gpu_thunk_name and \
          not manager.is_dependent(ndp_custom_call, gpu_custom_call) and \
          manager.get_custom_call_hops(ndp_custom_call) > manager.get_custom_call_hops(gpu_custom_call) :
        if overlap_gpu_thunk == '' \
            or manager.get_custom_call_hops(gpu_custom_call) > manager.get_custom_call_hops(custom_call_name(overlap_gpu_thunk)):
          overlap_gpu_thunk = gpu_custom_call
    if overlap_gpu_thunk == '':
      exit()
    scheduled_kernels[overlap_gpu_thunk].append(ndp_thunk_name)
    hops_map[overlap_gpu_thunk] = manager.get_custom_call_hops(ndp_custom_call)
    NDP_thunks.remove((order, ndp_thunk_name))
    for operand in manager.get_operands(ndp_custom_call):
      no_cxl_flags[operand] = False
      
"""
Step 2, Schedule Wegith update
"""
NDP_thunks_cpy = NDP_thunks.copy()
for order, ndp_thunk_name in NDP_thunks_cpy:
  ndp_custom_call = custom_call_name(ndp_thunk_name)
  ndp_hops = manager.get_custom_call_hops(ndp_custom_call)
  if 'Adam' in ndp_thunk_name:
    overlap_gpu_thunk = ''
    for gpu_order, gpu_thunk_name in GPU_thunks:
      gpu_custom_call = custom_call_name(gpu_thunk_name)
      gpu_hops = manager.get_custom_call_hops(gpu_custom_call) 
      if 'custom-call' in gpu_thunk_name and\
          not manager.is_dependent(ndp_custom_call, gpu_custom_call) and\
          ndp_hops <= gpu_hops and\
          not list_substr_contains(scheduled_kernels[gpu_custom_call], 'BackwardBertOutput') and \
          not list_substr_contains(scheduled_kernels[gpu_custom_call], 'SoftmaxBackward'):
        if overlap_gpu_thunk == ''\
            or (manager.get_custom_call_hops(gpu_custom_call) <= manager.get_custom_call_hops(custom_call_name(overlap_gpu_thunk)) \
            and len(scheduled_kernels[gpu_custom_call]) < len(scheduled_kernels[overlap_gpu_thunk])):
          overlap_gpu_thunk = gpu_custom_call
          print(f'NDP({ndp_thunk_name}) mapped to {gpu_thunk_name}')
          overlapped_candidates.append(gpu_thunk_name)

    scheduled_kernels[overlap_gpu_thunk].append(ndp_thunk_name)
    if overlap_gpu_thunk != '':
      hops_map[overlap_gpu_thunk] = manager.get_custom_call_hops(overlap_gpu_thunk)  - 0.5
     
    NDP_thunks.remove((order, ndp_thunk_name))

"""
Step 3, cost model overlapping
warning: there might be ndp kernels that do not have overlapping kernel
NDPX trace format: _NDP_custom-call.149_0_NdpEwiseFused$fusion.14.traceg
"""
NDP_thunks_cpy = NDP_thunks.copy()
for order, ndp_thunk_name in NDP_thunks_cpy:
  ndp_custom_call = custom_call_name(ndp_thunk_name)
  ndp_hops = manager.get_custom_call_hops(ndp_custom_call)
  if "NdpEwiseFused" in ndp_thunk_name:
    print(f"ndp_thunk_name: {ndp_thunk_name}")
    # the custom call target itself do not have overlapping information, so find the 
    # overlapping target from the file.
    # file format: _ON_THE_FLY_custom-call.0_0_NdpEwiseFusedOnTheFly$fusion.21.traceg
    # file format: _NDP_custom-call.163_0_NdpEwiseFused$custom-call.69.traceg
    for trace_file in ndpx_trace_files:
      if (ndp_custom_call in trace_file) and ('adam' not in trace_file):
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
            break
  

for key in manager.get_hops_map():
  manager.refill_hops_map(hops_map, key)

with open(output+'.hops', 'w') as f_hops:
  f_hops.write(f'ID,HOPS\n')
  with open(output+'.bw', 'w') as f_bw:
    with open(output+'.fw', 'w') as f_fw:
      for (order, thunk_name), kernels in matched:
        gpu_custom_call = custom_call_name(thunk_name)
        hops = manager.get_custom_call_hops(gpu_custom_call)
        if (hops > args.fw_hops):
          f = f_bw
        else:
          f = f_fw
        # use metadata table for certifying whether it is on the right pass
        if gpu_custom_call in manager.metadata_table and \
            "gradient" in manager.metadata_table[gpu_custom_call]:
          f = f_bw
        for kernel_no, kernel_name in kernels:
          f.write(f'// Thunk: {thunk_name}\n')
          f.write(f'// Kernel Name: {kernel_name}\n')
          f.write(f'// Hops: {hops}\n')
          f.write(f'// Order: {hops_map[gpu_custom_call] * 100000 + kernel_no}\n')
          if no_cxl_flags[gpu_custom_call]:
            f.write(f'# NO_CXL\n')
          adam_used = False
          for ndp_thunk_name in scheduled_kernels[gpu_custom_call]:
            if 'Adam' in ndp_thunk_name:
              adam_used = True
              f.write(f'_NDP_{ndp_thunk_name.split(":")[0]}_adam_reduce.traceg\n')
          if adam_used:
            f.write(f'_BAR_\n')
          for ndp_thunk_name in scheduled_kernels[gpu_custom_call]:
            # print(ndp_thunk_name)
            if 'Adam' in ndp_thunk_name:
              f.write(f'_NDP_{ndp_thunk_name.split(":")[0]}_adam.traceg\n')
            elif 'BackwardBertOutput' in ndp_thunk_name:
              f.write(f'_NDP_{ndp_thunk_name.split(":")[0]}_bw_bert_output_ph1.traceg\n')
              f.write(f'_BAR_\n')
              f.write(f'_NDP_{ndp_thunk_name.split(":")[0]}_bw_bert_output_ph2.traceg\n')
              f.write(f'_BAR_\n')
              f.write(f'_NDP_{ndp_thunk_name.split(":")[0]}_bw_bert_output_ph3.traceg\n')
            elif 'SoftmaxBackward' in ndp_thunk_name:
              f.write(f'_NDP_{ndp_thunk_name.split(":")[0]}_bw_bert_softmax_ph1.traceg\n')
              f.write(f'_BAR_\n')
              f.write(f'_NDP_{ndp_thunk_name.split(":")[0]}_bw_bert_softmax_reduce_accum_1.traceg\n')
              f.write(f'_BAR_\n')
              f.write(f'_NDP_{ndp_thunk_name.split(":")[0]}_bw_bert_softmax_reduce_accum_2.traceg\n')
              f.write(f'_BAR_\n')
              f.write(f'_NDP_{ndp_thunk_name.split(":")[0]}_bw_bert_softmax_mul.traceg\n')
            elif 'NdpEwiseFused' in ndp_thunk_name:
              ndp_custom_call = custom_call_name(ndp_thunk_name)
              # make on-the-fly trace files to be written.
              # TODO: multiple scheduling + _ON_THE_FLY_ handling
              for trace_file in ndpx_trace_files:
                if ('NdpEwiseFused' in trace_file) and (ndp_custom_call in trace_file):
                  f.write(f'{trace_file}\n')
                  # if '_ON_THE_FLY' in trace_file:
                  #   f.write(f'_BAR_\n')
            else:
              f.write(f'{ndp_thunk_name}\n')
          f.write(f'kernel-{kernel_no}.traceg\n')
          f.write('\n')
          f_hops.write(f'{kernel_no},{hops}\n')
        if 'custom-call' in thunk_name:
          num_gpu_custom_call -= 1
        finish_list.append(thunk_name.split(':')[0])
      first = True
      f = f_bw
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

# print(f'Remaining ndp thunks {NDP_thunks}')
# print(num_gpu_custom_call)

for order2, ndp_thunk_name in NDP_thunks:
  # print('NAME: ' + ndp_thunk_name)
  manager.print_unfinished_parent(ndp_thunk_name.split(":")[0])
