import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help="Model name", required=True)

def copytree(src, dst, symlinks=False, ignore=None):
  for item in os.listdir(src):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if os.path.isdir(s):
      shutil.copytree(s, d, symlinks, ignore)
    else:
      shutil.copy2(s, d)

if __name__ == "__main__":
  args = parser.parse_args()
  tracedir_path = f'/home/jueonpark/tracegen/traces/{args.model}/traces/'
  fw_kernelslist_path = f'/home/jueonpark/tracegen/traces/{args.model}/kernelslist.g.fw'
  bw_kernelslist_path = f'/home/jueonpark/tracegen/traces/{args.model}/kernelslist.g.bw'
  fw_tracedir_path = f'/home/jueonpark/tracegen/traces/{args.model}/traces_fw/'
  bw_tracedir_path = f'/home/jueonpark/tracegen/traces/{args.model}/traces_bw/'
  xlahlo_path = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo/'
  fw_xlahlo_path = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo_fw/'
  bw_xlahlo_path = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo_bw/'
  fw_kernelslist = open(fw_kernelslist_path, 'r').read().split('\n\n')
  bw_kernelslist = open(bw_kernelslist_path, 'r').read().split('\n\n')

  # generate directories
  if not os.path.exists(fw_tracedir_path):
    os.makedirs(fw_tracedir_path)
    print("Directory ", fw_tracedir_path, " Created ")
  else:    
    print("Directory ", fw_tracedir_path, " already exists")   
  if not os.path.exists(bw_tracedir_path):
    os.makedirs(bw_tracedir_path)
    print("Directory ", bw_tracedir_path, " Created ")
  else:    
    print("Directory ", bw_tracedir_path, " already exists")   
  # copy
  if not os.path.exists(fw_xlahlo_path):
    copytree(xlahlo_path, fw_xlahlo_path)
    print("Directory ", fw_xlahlo_path, " Copied ")
  else:    
    print("Directory ", fw_xlahlo_path, " already exists")
  if not os.path.exists(bw_xlahlo_path):
    copytree(xlahlo_path, bw_xlahlo_path)
    print("Directory ", bw_xlahlo_path, " Copied ")
  else:    
    print("Directory ", bw_xlahlo_path, " already exists")   

  # move files
  print("fw")
  for kernel in fw_kernelslist:
    # Get the overlapping information from kernelslist.g
    # Thunk: custom-call.6:__cublas$gemm
    # Kernel Name: turing_fp16_s884gemm_fp16_64x64_ldg8_f2f_nn
    kernelfile = kernel.split('\n')[-1]
    print(kernelfile)
    prev_path = os.path.join(tracedir_path, kernelfile)
    if prev_path == tracedir_path:
      print(kernel)
      continue
    move_path = os.path.join(fw_tracedir_path, kernelfile)
    try:
      shutil.move(prev_path, move_path)
    except:
      print(prev_path)

  print("bw")
  for kernel in bw_kernelslist:
    # Get the overlapping information from kernelslist.g
    # Thunk: custom-call.6:__cublas$gemm
    # Kernel Name: turing_fp16_s884gemm_fp16_64x64_ldg8_f2f_nn
    kernelfile = kernel.split('\n')[-1]
    print(kernelfile)
    prev_path = os.path.join(tracedir_path, kernelfile)
    if prev_path == tracedir_path:
      print(kernel)
      continue
    move_path = os.path.join(bw_tracedir_path, kernelfile)
    try:
      shutil.move(prev_path, move_path)
    except:
      print(prev_path)
  
  print("kernelslists")
  shutil.move(fw_kernelslist_path, fw_tracedir_path)
  shutil.move(bw_kernelslist_path, bw_tracedir_path)