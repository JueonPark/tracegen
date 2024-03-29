import os 
import argparse

from utils2 import parse_thunk_schedule
from utils2 import parse_stats
from utils2 import HloDepdendencyManager

def match(ts_parsed, stats_parsed):
  # while True:
  #   if ts_parsed[0] == stats_parsed[0][1]:
  #     break
  #   ts_parsed = ts_parsed[1:]

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

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--stats-path', type=str, help="stats.csv file path", required=True)
parser.add_argument('-t', '--ts-path', type=str, help="thunkschedule file path", required=True)
parser.add_argument('-g', '--graph-path', type=str, help="hlo after optimizations txt file path", required=True)
parser.add_argument('-o', '--output', type=str, help="output file path", required=True)
parser.add_argument('-S', '--start', type=int, help="kernel start number", required=True)
parser.add_argument('-E', '--end', type=int, help="kernel end number", default=-1)
parser.add_argument('-f', '--fw-hops', type=int, help="forward kernel hops", default=12)

args = parser.parse_args()

GPU_thunks, NDP_thunks = parse_thunk_schedule(open(args.ts_path).read())
# print(GPU_thunks)
stats_parsed = parse_stats(open(args.stats_path).read(), args.start, args.end)
manager = HloDepdendencyManager(open(args.graph_path).read())
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
    # print(f'GPU {thunk} : {manager.get_custom_call_hops(custom_call_name(thunk))}')
    # manager.print_direct_parent_custom_call(thunk.split(':')[0])    

# for order, thunk in NDP_thunks:
  # if 'custom-call' in thunk:
    # print(f'NDP {thunk} : {manager.get_custom_call_hops(custom_call_name(thunk))}')
    # manager.print_direct_parent_custom_call(thunk.split(':')[0])
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
      if 'custom-call' in gpu_thunk_name and\
          manager.is_dependent(ndp_custom_call, gpu_custom_call) and\
          ndp_hops == gpu_hops + 1:
          if ph1_used:
            print("ERROR")
            exit()
          no_cxl_flags[gpu_custom_call] = False
          scheduled_kernels[gpu_custom_call].append(f'_ON_THE_FLY_{ndp_custom_call}_fw_bert_softmax_ph1.traceg')
          ph1_used = True
      elif 'custom-call' in gpu_thunk_name and\
          not manager.is_dependent(ndp_custom_call, gpu_custom_call) and\
          ndp_hops == gpu_hops + 2:
          if ph3_used:
            print("ERROR")
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
      if 'custom-call' in gpu_thunk_name and\
          manager.is_dependent(ndp_custom_call, gpu_custom_call) and\
          ndp_hops == gpu_hops + 1:
          no_cxl_flags[gpu_custom_call] = False
          scheduled_kernels[gpu_custom_call].append(f'_ON_THE_FLY_{ndp_custom_call}_fw_bert_output_ph1.traceg')
          scheduled_kernels[gpu_custom_call].append(f'_BAR_')
          scheduled_kernels[gpu_custom_call].append(f'_NDP_{ndp_custom_call}_fw_bert_output_ph2_mean.traceg')
          scheduled_kernels[gpu_custom_call].append(f'_BAR_')
          scheduled_kernels[gpu_custom_call].append(f'_NDP_{ndp_custom_call}_fw_bert_output_ph2_var.traceg')
          if args.fw_hops - ndp_hops <= 2:
            scheduled_kernels[gpu_custom_call].append(f'_BAR_')
            scheduled_kernels[gpu_custom_call].append(f'_NDP_{ndp_custom_call}_fw_bert_output_ph3.traceg')
            ph3_used = True

          if ph1_used:
            print("ERROR")
            exit()
          ph1_used = True
      elif 'custom-call' in gpu_thunk_name and\
          manager.is_dependent(gpu_custom_call, ndp_custom_call) and\
          ndp_hops + 1 == gpu_hops and not ph3_used:
          scheduled_kernels[gpu_custom_call].append(f'_ON_THE_FLY_{ndp_custom_call}_fw_bert_output_ph3.traceg')
          ph3_used = True
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
      if 'custom-call' in gpu_thunk_name and\
          not manager.is_dependent(ndp_custom_call, gpu_custom_call) and\
          manager.get_custom_call_hops(ndp_custom_call) > manager.get_custom_call_hops(gpu_custom_call) :
        if overlap_gpu_thunk == '' \
            or manager.get_custom_call_hops(gpu_custom_call) > manager.get_custom_call_hops(custom_call_name(overlap_gpu_thunk)):
          overlap_gpu_thunk = gpu_custom_call
    if overlap_gpu_thunk == '':
      exit
    scheduled_kernels[overlap_gpu_thunk].append(ndp_thunk_name)
    hops_map[overlap_gpu_thunk] = manager.get_custom_call_hops(ndp_custom_call)
    NDP_thunks.remove((order, ndp_thunk_name))
    for opernd in manager.get_operands(ndp_custom_call):
      no_cxl_flags[opernd] = False
      
  

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

    scheduled_kernels[overlap_gpu_thunk].append(ndp_thunk_name)
    if overlap_gpu_thunk != '':
      hops_map[overlap_gpu_thunk] = manager.get_custom_call_hops(overlap_gpu_thunk)  - 0.5
     

    NDP_thunks.remove((order, ndp_thunk_name))

for key in manager.get_hops_map():
  manager.refill_hops_map(hops_map, key)

with open(args.output+'.hops', 'w') as f_hops:
  f_hops.write(f'ID,HOPS\n')
  with open(args.output+'.bw', 'w') as f_bw:
    with open(args.output+'.fw', 'w') as f_fw:
      for (order, thunk_name), kernels in matched:
        gpu_custom_call = custom_call_name(thunk_name)
        hops = manager.get_custom_call_hops(gpu_custom_call)
        if( hops > args.fw_hops):
          f = f_bw
        else:
          f = f_fw
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

print(f'Remaining ndp thunks {NDP_thunks}')
print(num_gpu_custom_call)

for order2, ndp_thunk_name in NDP_thunks:
  print('NAME: ' + ndp_thunk_name)
  manager.print_unfinished_parent(ndp_thunk_name.split(":")[0])