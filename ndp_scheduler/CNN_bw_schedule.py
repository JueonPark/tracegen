from asyncore import write
import os
import argparse
from tracemalloc import start

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--tspath', type=str, required=True)
parser.add_argument('--statpath', type=str, required=True)
parser.add_argument('--bw_start', type=int, required=True)
args = parser.parse_args()

model_home = args.model
# NDP_kernel_home = f'{model_home}/xla_hlo_fw/packet_32_buffer_8_gpu_8_sync_0_simd_8/'
# NDP_kernels = [i for i in os.listdir(NDP_kernel_home) if '_NDP_' in i or '_ON_THE_' in i]

thunk_schedule_file = open(args.tspath, 'r')
thunk_schedule = thunk_schedule_file.readlines()
thunk_schedule_file.close()

stat_file = open(args.statpath, 'r')
stat = stat_file.readlines()
stat_file.close()

# kernelslist_file = open(f'{args.model}/traces/kernelslist.g', 'r')
# kernelslist = kernelslist_file.readlines()
# kernelslist_file.close()

# setup start
start_kernel_name = ""
start_thunk_name = ""
stat_start_idx = 0
for i, s in enumerate(stat):
   if f'kernel-{args.bw_start}' in s:
      start_kernel_name = s.split(',')[1][1:]
      start_thunk_name = '.'.join(start_kernel_name.split('_'))
      stat_start_idx = i
      break

thunk_start_idx = 0
for i, thunk in enumerate(thunk_schedule):
   thunk_name = start
   if start_thunk_name in thunk:
      start_idx = i
      break
# setup end

ndp_conv_custom_calls = []
slice_ndp_conv_custom_calls = []
dgrad_pool = []
adam_pool = [[f'_BAR_\n']]

unscheduled = []

for i in range(thunk_start_idx, len(thunk_schedule)):
   line = thunk_schedule[i]
   name = line.split('\t')[-1][:-1]
   if 'ConvBackward' in name or 'ConvFirstLayerBackward' in name:
      custom_call_id = int(name.split('.')[1].split(':')[0])
      NDP_kernels = [f'_NDP_custom-call.{custom_call_id}_bw_bn_ph1.traceg\n',
         f'_BAR_\n',
         f'_NDP_custom-call.{custom_call_id}_bw_bn_ph2.traceg\n',
         f'_BAR_\n',
         f'_NDP_custom-call.{custom_call_id}_bw_bn_ph3.traceg\n',
         f'_BAR_\n',
         f'_NDP_custom-call.{custom_call_id}_bw_bias.traceg\n',
         f'_BAR_\n',
         f'_NDP_custom-call.{custom_call_id}_bw_bias_reduce.traceg\n']
      dgrad_pool.append(NDP_kernels)
      ndp_conv_custom_calls.append(custom_call_id)
      adam_pool.append([f'_BAR_\n'])
      if 'slice' in thunk_schedule[i-1]:
         slice_ndp_conv_custom_calls.append(custom_call_id)
   elif '$Gemm' in name:
      if 'get-tuple-element' in name:
         continue
      custom_call_id = int(name.split('.')[1].split(':')[0])
      adam_pool[0] = [f'_NDP_custom-call.{custom_call_id}_adam_reduce.traceg\n']+adam_pool[0]+[f'_NDP_custom-call.{custom_call_id}_adam.traceg\n']
   elif 'Adam' in name and 'Gemm' not in name:
      # print(name)
      custom_call_id = int(name.split('.')[1].split(':')[0])
      if len(name.split('.')) < 3:
         unscheduled.append([f'_NDP_custom-call.{custom_call_id}_adam_reduce.traceg\n'])
         unscheduled.append([f'_NDP_custom-call.{custom_call_id}_adam.traceg\n'])
         continue
      conv_id = int(name.split('.')[-1])
      idx = ndp_conv_custom_calls.index(conv_id)+1
      adam_pool[idx] = [f'_NDP_custom-call.{custom_call_id}_adam_reduce.traceg\n']+adam_pool[idx]+[f'_NDP_custom-call.{custom_call_id}_adam.traceg\n']

print(unscheduled)

assert(len(dgrad_pool)+1 == len(adam_pool))
num_ndp_conv_custom_calls = len(ndp_conv_custom_calls)
print(f'len(dgrad_pool): {len(dgrad_pool)}')
print(f'len(adam_pool): {len(adam_pool)}')
print(f'num_ndp_conv_custom_calls: {num_ndp_conv_custom_calls}')
# print(f'dgrad_pool: {dgrad_pool}')
# print(f'adam_pool: {adam_pool}')

def get_kernel_type(kernel_name):
   if 'dgrad' in kernel_name or '_tn_' in kernel_name:
      return 'dgrad'
   elif 'first_layer_wgrad' in kernel_name or 'wgrad_alg0_engine_NHWC' in kernel_name:
      return 'first_wgrad'
   elif 'wgrad' in kernel_name and 'reduction' not in kernel_name:
      return 'wgrad'
   elif 'turing' in kernel_name and 'gemm' in kernel_name:
      return 'gemm'
   elif 'slice' in kernel_name:
      return 'slice'
   elif 'broadcast' in kernel_name:
      return 'broadcast'
   elif 'copy' in kernel_name:
      return 'copy'
   else:
      return 'others'

dgrad_count = 0
wgrad_count = 0
gemm_count = 0
first_wgrad_met = False

output_file = open(f'{model_home}/traces_bw/kernelslist.g.bw', 'w')
output_buffer = []

def get_latest_index(kernel_type):
   for i in range(len(output_buffer)-1, -1, -1):
      for j in output_buffer[i]:
         if kernel_type in j:
            return i
   print(f'output_buffer[0][0]: {output_buffer[0][0]}')
   print(f'output_buffer[0]: {output_buffer[0]}')
   print(f'kernel_type: {kernel_type}')
   print(f'output_buffer: {output_buffer}')
   assert False

first_dgrad_kernels = dgrad_pool[0]
dgrad_pool = dgrad_pool[1:]

for i in range(stat_start_idx, len(stat)):
   write_to_NDP = False
   words = stat[i].split(',')
   file_name = f'{words[0]}g'
   kernel_name = words[1][1:]
   schedule = []
   idx = -1
   kernel_type = get_kernel_type(kernel_name)
   if kernel_type == 'dgrad': # overlapped with adam kernels
      ndp_kernels_to_overlap = adam_pool[0]
      schedule.append(f'// custom-call.{ndp_conv_custom_calls[0]}\n')
      for ndp_kernel in ndp_kernels_to_overlap:
         schedule.append(ndp_kernel)
      adam_pool = adam_pool[1:]
      if dgrad_count < num_ndp_conv_custom_calls-2 and ndp_conv_custom_calls[0] not in slice_ndp_conv_custom_calls:
         write_to_NDP = True
      dgrad_count+=1
      if dgrad_count != wgrad_count+1:
         print(kernel_name)
         assert False
      # print(f'dgrad_count: {dgrad_count}')
   elif kernel_type == 'wgrad' or kernel_type == 'first_wgrad': # overlapped with bn kernels
      if 'first' in kernel_type: # first wgrad, exceptionally overlapped with adam kernels
         ndp_kernels_to_overlap = adam_pool[0]
         schedule.append(f'// custom-call.{ndp_conv_custom_calls[0]}\n')
         for ndp_kernel in ndp_kernels_to_overlap:
            schedule.append(ndp_kernel)
         adam_pool = adam_pool[1:]
         first_wgrad_met = True
      else: # other wgrads, overlapped with dgrad kernels
         if len(dgrad_pool) < 1:
            print(kernel_name)
         ndp_kernels_to_overlap = dgrad_pool[0]
         schedule.append(f'// custom-call.{ndp_conv_custom_calls[0]}\n')
         for ndp_kernel in ndp_kernels_to_overlap:
            schedule.append(ndp_kernel)
         dgrad_pool = dgrad_pool[1:]
      ndp_conv_custom_calls = ndp_conv_custom_calls[1:]
      wgrad_count+=1
   elif kernel_type == 'gemm':
      if gemm_count == 1: # wgrad_gemm, overlapped with first dgrad kernels
         ndp_kernels_to_overlap = first_dgrad_kernels
         for ndp_kernel in ndp_kernels_to_overlap:
            schedule.append(ndp_kernel)
         idx = get_latest_index('broadcast')
      else:
         print()
      gemm_count+=1
   elif kernel_type == 'slice':
      write_to_NDP = True
      idx = get_latest_index('wgrad')-1
   elif kernel_type == 'broadcast':
      write_to_NDP = True
   else: # other types of kernels. e.g., fusion, pad ...
      if first_wgrad_met:
         if len(adam_pool) >= 2:
            print(f'dgrad_count: {dgrad_count}')
            print(f'wgrad_count: {wgrad_count}')
            print(f'gemm_count: {gemm_count}')

            print(adam_pool)
         assert(len(adam_pool) < 2)
         if len(adam_pool) > 0:
            ndp_kernels_to_overlap = adam_pool[0]
            for ndp_kernel in ndp_kernels_to_overlap:
               schedule.append(ndp_kernel)
            adam_pool = adam_pool[1:]

   schedule.append(f'// kernel name: {kernel_name}\n')
   if not write_to_NDP:
      schedule.append(f'# NO_CXL\n')
   schedule.append(f'{file_name}\n')
   schedule.append('\n')
   if ('resnet50' in args.model or 'mobilenet' in args.model) and kernel_type == 'copy':
      continue
   if idx == -1:
      output_buffer.append(schedule)
   else:
      # print(f'output_buffer[:idx]: {output_buffer[:idx]}')
      # print(f'output_buffer[idx:]: {output_buffer[idx:]}')
      output_buffer = output_buffer[:idx+1]+[schedule]+output_buffer[idx+1:]

for schedule in output_buffer:
   for kernel in schedule:
      output_file.write(kernel)

output_file.close()
# print(output_buffer)

# output_file = open(f'{model_home}/traces_bw/kernelslist.g.bw', 'w')
# for i in range(stat_start_idx, len(stat)):
#    write_to_NDP = False
#    words = stat[i].split(',')
#    file_name = f'{words[0]}g'
#    kernel_name = words[1][1:]
#    kernel_type = get_kernel_type(kernel_name)
#    if kernel_type == 'dgrad': # overlapped with adam kernels
#       ndp_kernels_to_overlap = adam_pool[0]
#       output_file.write(f'// custom-call.{ndp_conv_custom_calls[0]}\n')
#       for ndp_kernel in ndp_kernels_to_overlap:
#          output_file.write(ndp_kernel)
#       adam_pool = adam_pool[1:]
#       if dgrad_count < num_ndp_conv_custom_calls-2 and ndp_conv_custom_calls[0] not in slice_ndp_conv_custom_calls:
#          write_to_NDP = True
#       dgrad_count+=1
#    elif kernel_type == 'wgrad' or kernel_type == 'first_wgrad': # overlapped with bn kernels
#       if 'first' in kernel_type: # first wgrad, exceptionally overlapped with adam kernels
#          ndp_kernels_to_overlap = adam_pool[0]
#          output_file.write(f'// custom-call.{ndp_conv_custom_calls[0]}\n')
#          for ndp_kernel in ndp_kernels_to_overlap:
#             output_file.write(ndp_kernel)
#          adam_pool = adam_pool[1:]
#          first_wgrad_met = True
#       else: # other wgrads, overlapped with dgrad kernels
#          ndp_kernels_to_overlap = dgrad_pool[0]
#          output_file.write(f'// custom-call.{ndp_conv_custom_calls[0]}\n')
#          for ndp_kernel in ndp_kernels_to_overlap:
#             output_file.write(ndp_kernel)
#          dgrad_pool = dgrad_pool[1:]
#       ndp_conv_custom_calls = ndp_conv_custom_calls[1:]
#       wgrad_count+=1
#    elif kernel_type == 'gemm':
#       if gemm_count == 1: # wgrad_gemm, overlapped with first dgrad kernels
#          ndp_kernels_to_overlap = dgrad_pool[0]
#          for ndp_kernel in ndp_kernels_to_overlap:
#             output_file.write(ndp_kernel)
#          dgrad_pool = dgrad_pool[1:]
#          write_to_NDP = True
#       gemm_count+=1
#    elif kernel_type == 'slice':
#       write_to_NDP = True
#    elif kernel_type == 'broadcast':
#       write_to_NDP = True
#    else: # other types of kernels. e.g., fusion, pad ...
#       if first_wgrad_met:
#          assert(len(adam_pool) < 2)
#          if len(adam_pool) > 0:
#             ndp_kernels_to_overlap = adam_pool[0]
#             for ndp_kernel in ndp_kernels_to_overlap:
#                output_file.write(ndp_kernel)
#             adam_pool = adam_pool[1:]

#    output_file.write(f'// kernel name: {kernel_name}\n')
#    if not write_to_NDP:
#       output_file.write(f'# NO_CXL\n')
#    output_file.write(f'{file_name}\n')
#    output_file.write('\n')
   
# output_file.close()