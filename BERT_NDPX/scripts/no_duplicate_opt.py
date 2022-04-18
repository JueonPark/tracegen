from posix import listdir
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--gpu', type=int, required=True)
args = parser.parse_args()

#models = ['vgg16_ndp_batch_16', 'vgg16_ndp_batch_32', 'resnet18_ndp_batch_16', 'resnet18_ndp_batch_64', 'resnet50_ndp_batch_16', 'resnet50_ndp_batch_64', 'mobilenetv2_ndp_batch_16', 'mobilenetv2_ndp_batch_64']

#models = ['new_resnet18_ndp_batch_16', 'new_resnet50_ndp_batch_16', 'new_mobilenetv2_ndp_batch_16']
models = [args.model]
gpus_list = [args.gpu]

#models = ['resnet18_ndp_batch_16', 'resnet18_ndp_batch_64', 'resnet50_ndp_batch_16', 'resnet50_ndp_batch_64']
#gpus_list = [2, 4]

home = '../traces/'
for model in models:
   model_home = home+model+'/'
   configs = os.listdir(model_home)
   for config in configs:
      if 'packet' not in config:
         continue
      words = config.split('_')
      gpus = int(words[5])
      direction = words[-1]
      if gpus in gpus_list and direction == 'bw':
         trace_home = model_home+config+'/'
         print('GPU:', gpus)
         kernels = os.listdir(trace_home)
         for kernel in kernels:
            kernel_home = trace_home+kernel+'/'
            kernelslists = os.listdir(kernel_home+'GPU_0/')
            is_adam_kernel = False
            for k in kernelslists:
               if 'adam' in k:
                  is_adam_kernel = True
            if is_adam_kernel == True:
               # print(kernel_home)
               gpu_folders = os.listdir(kernel_home)
               for gpu_folder in gpu_folders:
                  if 'GPU' not in gpu_folder:
                     continue
                  if int(gpu_folder.split('_')[-1]) > 0:
                     gpu_folder_to_modify = kernel_home+gpu_folder+'/'
                     print(gpu_folder_to_modify)
                     if 'kernelslist.g_bak' not in gpu_folders:
                        os.system('cp '+gpu_folder_to_modify+'kernelslist.g '+gpu_folder_to_modify+'kernelslist.g_bak')
                        os.system('rm '+gpu_folder_to_modify+'kernelslist.g')
                        old_list = open(gpu_folder_to_modify+'kernelslist.g_bak', 'r')
                        old_kernels = old_list.readlines()
                        old_list.close()
                        new_list = open(gpu_folder_to_modify+'kernelslist.g', 'w')
                        for line in old_kernels:
                           if 'kernel' in line:
                              new_list.write(line)
                        new_list.close()


