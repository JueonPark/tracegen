import os
from re import A
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--packet-size', type=int, required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--buffer', type=int, required=True)
parser.add_argument('--sync', type=int, required=True)
parser.add_argument('--simd', type=int, required=True)
args = parser.parse_args()

GPU_OFFSET = 0x1000000000000000

base_folder = f'./packet_32_buffer_4_gpu_4_sync_{args.sync}_simd_8'

if args.gpu == 1:
   base_folder = f'./packet_32_buffer_1_gpu_1_sync_{args.sync}_simd_8'

out_folder = f'./packet_{args.packet_size}_buffer_{args.buffer}_gpu_{args.gpu}_sync_{args.sync}_simd_8'

os.makedirs(out_folder, exist_ok=True)


def generate_ph2_mean(kernel_id, base, size, ld_cnt):
   metadata = []
   simd_0 = []

   metadata.append('-kernel name = _NDP_fw_bn_ph2_mean\n')
   metadata.append(f'-kernel id = {kernel_id}\n')
   metadata.append('\n')

   simd_0.append('SIMD 0\n')
   simd_0.append(f'SET_FILTER {hex(base)} {hex(size*2)} {size} -1\n')
   simd_0.append('CALL\n')
   for i in range(2):
      simd_0.append('MOVI.32 v0 00\n')
      for j in range(ld_cnt):
         simd_0.append(f'LDM.32 v{j+1} {hex(base)} FILTER\n')
      for j in range(ld_cnt):
         simd_0.append(f'ADD.32 v0 v0 v{j}\n')
      simd_0.append('MULTI.32 v0 v0 00\n')
      for j in range(ld_cnt):
         simd_0.append(f'STM.32 v0 {hex(base)} FILTER\n')
   simd_0.append('JOIN\n')
   simd_0.append('EXIT\n')
   simd_0.append('\n')
   return metadata, simd_0

def generate_ph2_var(kernel_id, base, size, ld_cnt):
   metadata = []
   simd_0 = []

   metadata.append('-kernel name = _NDP_fw_bn_ph2_var\n')
   metadata.append(f'-kernel id = {kernel_id}\n')
   metadata.append('\n')

   simd_0.append('SIMD 0\n')
   simd_0.append(f'SET_FILTER {hex(base)} {hex(size*2)} {size} -1\n')
   simd_0.append('CALL\n')
   for i in range(2):
      simd_0.append('MOVI.32 v0 00\n')
      for j in range(ld_cnt):
         simd_0.append(f'LDM.32 v{j+1} {hex(base)} FILTER\n')
      for j in range(ld_cnt):
         simd_0.append(f'ADD.32 v0 v0 v{j+1}\n')
      simd_0.append('MULTI.32 v0 v0 00\n')
      for j in range(ld_cnt):
         simd_0.append(f'STM.32 v0 {hex(base)} FILTER\n')
   simd_0.append('JOIN\n')
   simd_0.append('EXIT\n')
   simd_0.append('\n')
   return metadata, simd_0

def generate_ph2_bw(kernel_id, base, size, ld_cnt):
   metadata = []
   simd_0 = []

   metadata.append('-kernel name = _NDP_bw_bn_ph2\n')
   metadata.append(f'-kernel id = {kernel_id}\n')
   metadata.append('\n')

   simd_0.append('SIMD 0\n')
   simd_0.append(f'SET_FILTER {hex(base)} {hex(size*2)} {size} -1\n')
   simd_0.append('CALL\n')
   for i in range(4):
      simd_0.append('MOVI.32 v0 00\n')
      for j in range(ld_cnt):
         simd_0.append(f'LDM.32 v{j+1} {hex(i*size + base)} FILTER\n')
      for j in range(ld_cnt):
         simd_0.append(f'ADD.32 v0 v0 v{j+1}\n')
      for j in range(ld_cnt):
         simd_0.append(f'STM.32 v0 {hex(i*size + base)} FILTER\n')
   simd_0.append('JOIN\n')
   simd_0.append('EXIT\n')
   simd_0.append('\n')
   return metadata, simd_0

def generate_reduce(kernel_id, base, size, dim, axis, num_gpu, grad_addr, store_addr):
   metadata = []
   simd_0 = []

   metadata.append('-kernel name = _NDP_adam_reduce\n')
   metadata.append(f'-kernel id = {kernel_id}\n')
   metadata.append('\n')

   simd_0.append('SIMD 0\n')
   simd_0.append(f'SET_FILTER {hex(base)} {hex(size)} {dim} {axis}\n')
   simd_0.append('CALL\n')
   for g in range(num_gpu):
      simd_0.append(f'LDG.32 v{g+2} {hex(grad_addr + g*GPU_OFFSET)} OFFSET\n')
   for g in range(num_gpu-1):
      simd_0.append(f'ADD.32 v2 v2 v{g+3}\n')
   simd_0.append(f'MULTI.32 v2 v2 00\n')
   simd_0.append(f'STM.32 v2 BASE OFFSET\n')
   simd_0.append(f'STM.32 v2 {hex(store_addr)} OFFSET\n')
   simd_0.append('JOIN\n')
   simd_0.append('EXIT\n')
   simd_0.append('\n')



   return metadata, simd_0

def generate_broadcast(kernel_id, base, size, num_gpu):
   return metadata, simd_0



# copy all the files
file_list = os.listdir(base_folder)
for file_name in file_list:
   if 'ph2' not in file_name:
      shutil.copy(base_folder+'/'+file_name, out_folder+'/'+file_name)

# modify ndp trace files
for file_name in file_list:
   # skip other files
   if 'NDP' not in file_name and 'ON_THE_FLY' not in file_name:
      continue
   ndp_file = open(base_folder+'/'+file_name, 'r')
   lines = ndp_file.readlines()
   ndp_file.close()
   metadata = []
   simd_0 = []
   simd_1 = []
   for line in lines:
      if 'SIMD' in line:
         break
      else:
         metadata.append(line)
   simd_start = 0
   for line in lines:
      if 'SIMD' in line:
         simd_start+=1
      if simd_start == 1:
         simd_0.append(line)
      elif simd_start == 2:
         simd_1.append(line)
      elif simd_start > 2:
         break
   
   # sanity check to make sure all the instructions for each SIMD is identical
   if len(simd_0) != len(simd_1):
      assert(False)
   for i, line in enumerate(simd_0):
      if i > 0 and (line != simd_1[i]):
         assert(False)      

   # modify simd instructions
   # Multiple filters for sync-bn
   set_filter_met = False
   if ('bn_ph1' in file_name or 'bn_ph3' in file_name) and args.sync == 1:
      new_simd_0  = []
      for line in simd_0:
         if 'SET_FILTER' not in line:
            new_simd_0.append(line)
         else:
            if set_filter_met:
               continue 
            for i in range(args.gpu):
               words = line.split(' ')
               words[1] = hex(int(words[1], 16) + i*GPU_OFFSET)
               new_filter = ' '.join(words)
               new_simd_0.append(new_filter)
            set_filter_met = True
      simd_0 = new_simd_0

   # Address alignment
   base_offset = 0
   for i, line in enumerate(simd_0):
      if 'SET_FILTER' in line:
         base_addr = int(line.split(' ')[1], 16)
         base_offset = base_addr % (args.packet_size * args.simd * args.buffer)
      if 'LD' in line or 'ST' in line:
         if 'BASE' in line:
            continue
         words = line.split(' ')
         addr = int(words[2], 16)
         round = int(addr / (args.packet_size * args.simd * args.buffer))
         new_addr = round * (args.packet_size * args.simd * args.buffer) + base_offset
         words[2] = hex(new_addr)
         simd_0[i] = ' '.join(words)

   out_file = open(out_folder+'/'+file_name, 'w')
   for line in metadata:
      out_file.write(line)
   for i in range(args.simd * args.buffer):
      for line in simd_0:
         if 'SIMD' in line:
            new_line = f'SIMD {i}\n'
            out_file.write(new_line)
         else:
            out_file.write(line)
   out_file.close()

for file_name in file_list:   
   # Additional ph2 if required
   if 'fw_bn_ph1' in file_name:
      custom_call_number = int(file_name.split('.')[1].split('_')[0])
      mean_file_name = f'_NDP_custom-call.{custom_call_number}_fw_bn_ph2_mean.traceg'
      var_file_name = f'_NDP_custom-call.{custom_call_number}_fw_bn_ph2_var.traceg'
      # if os.path.isfile(out_folder+'/'+mean_file_name):
      #    continue
      # else:
      #    print(out_folder+'/'+mean_file_name)
      #    assert(False)
      ph1_file = open(out_folder+'/'+file_name, 'r')
      ph1_traces = ph1_file.readlines()

      kernel_id = 0
      base_mean = 0
      base_var = 0
      size = 0

      for line in ph1_traces:
         if 'kernel id' in line:
            kernel_id = int(line.split(' ')[-1][:-1])
         if 'SET_FILTER' in line and size == 0:
            size = int(line.split(' ')[-2].split(',')[-1])
            ld_cnt = int(args.buffer * args.simd / (size/(args.packet_size//2)))
         if 'STM' in line and 'v0' in line and base_mean == 0:
            base_mean = int(line.split(' ')[2], 16)
         if 'STM' in line and 'v1' in line and base_var == 0:
            base_var = int(line.split(' ')[2], 16)

      metadata, simd_0 = generate_ph2_mean(kernel_id, base_mean, size, ld_cnt)
      mean_file = open(out_folder+'/'+mean_file_name, 'w')
      for line in metadata:
         mean_file.write(line)
      for i in range(args.simd * args.buffer):
         for line in simd_0:
            if 'SIMD' in line:
               new_line = f'SIMD {i}\n'
               mean_file.write(new_line)
            else:
               mean_file.write(line)
      mean_file.close()

      metadata, simd_0 = generate_ph2_var(kernel_id, base_var, size, ld_cnt)
      var_file = open(out_folder+'/'+var_file_name, 'w')
      for line in metadata:
         var_file.write(line)
      for i in range(args.simd * args.buffer):
         for line in simd_0:
            if 'SIMD' in line:
               new_line = f'SIMD {i}\n'
               var_file.write(new_line)
            else:
               var_file.write(line)
      var_file.close()

      
   if 'bw_bn_ph1' in file_name:
      custom_call_number = int(file_name.split('.')[1].split('_')[0])
      ph2_file_name = f'_NDP_custom-call.{custom_call_number}_bw_bn_ph2.traceg'
      # if os.path.isfile(out_folder+'/'+ph2_file_name):
      #    continue
      # else:
      #    print(out_folder+'/'+ph2_file_name)
      #    assert(False)
      ph1_file = open(out_folder+'/'+file_name, 'r')
      ph1_traces = ph1_file.readlines()

      kernel_id = 0
      base = 0
      size = 0
      ph2_required = False

      for line in ph1_traces:
         if 'kernel id' in line:
            kernel_id = int(line.split(' ')[-1][:-1])
         if 'SET_FILTER' in line and size == 0:
            size = int(line.split(' ')[-2].split(',')[-1])
            ld_cnt = int(args.buffer * args.simd / (size/(args.packet_size//2)))
            if args.buffer * args.simd > (size/(args.packet_size//2)):
               ph2_required = True
         if 'STM' in line and 'FILTER' in line and base == 0:
            base = int(line.split(' ')[2], 16)
      
      if not ph2_required:
         continue

      metadata, simd_0 = generate_ph2_bw(kernel_id, base, size, ld_cnt)
      ph2_file = open(out_folder+'/'+ph2_file_name, 'w')
      for line in metadata:
         ph2_file.write(line)
      for i in range(args.simd * args.buffer):
         for line in simd_0:
            if 'SIMD' in line:
               new_line = f'SIMD {i}\n'
               ph2_file.write(new_line)
            else:
               ph2_file.write(line)
      ph2_file.close()

   if 'adam_reduce' in file_name:
      # print(out_folder+'/'+file_name)
      if args.gpu == 1:
         assert(False)
      
      is_STG = False
      kernel_id = 0
      base = 0
      size = 0
      dim = 0
      axis = 0
      grad_addr = 0
      store_addr = 0

      trace_file = open(out_folder+'/'+file_name, 'r')
      trace_lines = trace_file.readlines()
      trace_file.close()
      for trace_line in trace_lines:
         if 'SET_FILTER' in trace_line:
            words = trace_line.split(' ') 
            base = int(words[1], 16)
            size = int(words[2], 16)
            dim = int(words[3])
            axis = -1
         if 'kernel id' in trace_line:
            kernel_id = int(trace_line.split(' ')[-1][:-1])
         if 'LDG' in trace_line:
            is_STG = True
            if grad_addr == 0:
               grad_addr = int(trace_line.split(' ')[2], 16)
         if 'STM' in trace_line:
            if 'BASE' not in trace_line:
               store_addr = int(trace_line.split(' ')[2], 16)

      if is_STG:
         metadata, simd_0 = generate_reduce(kernel_id, base, size, dim, axis, args.gpu, grad_addr, store_addr)
         trace_file = open(out_folder+'/'+file_name, 'w')
         for line in metadata:
            trace_file.write(line)
         for i in range(args.simd * args.buffer):
            for line in simd_0:
               if 'SIMD' in line:
                  new_line = f'SIMD {i}\n'
                  trace_file.write(new_line)
               else:
                  trace_file.write(line)
         trace_file.close()
   
   if 'adam.traceg' in file_name:
      if args.gpu == 1:
         continue

      is_STGG = False
      store_addr = 0

      metadata = []
      simd_0 = []

      simd_met = False

      trace_file = open(out_folder+'/'+file_name, 'r')
      trace_lines = trace_file.readlines()
      trace_file.close()
      for trace_line in trace_lines:
         if simd_met == False:
            metadata.append(trace_line)
         if 'SIMD 0' in trace_line:
            simd_met = True
            simd_0.append(trace_line)
         if simd_met:
            simd_0.append(trace_line)
         if 'STGG' in trace_line:
            is_STGG = True
            if store_addr == 0:
               store_addr = int(trace_line.split(' ')[2], 16)
               break

      if is_STGG:
         for g in range(1, args.gpu):
            simd_0.append(f'STGG.32 v5 {hex(store_addr + g*GPU_OFFSET)} OFFSET\n')
         simd_0.append('JOIN\n')
         simd_0.append('EXIT\n\n')

         trace_file = open(out_folder+'/'+file_name, 'w')
         for line in metadata:
            trace_file.write(line)
         for i in range(args.simd * args.buffer):
            for line in simd_0:
               if 'SIMD' in line:
                  new_line = f'SIMD {i}\n'
                  trace_file.write(new_line)
               else:
                  trace_file.write(line)
         trace_file.close()