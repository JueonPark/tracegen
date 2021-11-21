#!/bin/bash

# batch 16, 4 GPU, sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 4 4 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 4 4 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 4 4 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-sync

# batch 16, 4 GPU, no sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 4 4 0 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 4 4 0 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 4 4 0 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-nosync

# batch 16, 2 GPU, sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 2 2 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 2 2 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 2 2 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-sync

# batch 16, 2 GPU, no sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 2 2 0 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 2 2 0 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 2 2 0 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-nosync

# batch 16, 1 GPU, no sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 1 1 0 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 1 1 0 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 1 1 0 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-1-nosync

# batch 64
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_64 4 4 1 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_64 4 4 0 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_64 2 2 1 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_64 2 2 0 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_64 1 1 0 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync


#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_64 4 4 1 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_64 4 4 0 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_64 2 2 1 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_64 2 2 0 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_64 1 1 0 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync




#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_64 4 4 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_64 4 4 0 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_64 2 2 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_64 2 2 0 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_64 1 1 0 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync


#Link 1.5x
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1.5 resnet50_ndp_batch_16 1 1 0 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1.5-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1.5 mobilenetv2_ndp_batch_16 1 1 0 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1.5-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1.5 vgg16_ndp_batch_16 1 1 0 > vgg16_ndp_batch_16-V100_downscaled_HBM2_PCI6x1.5-1-nosync

#Link 2x
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x2 resnet50_ndp_batch_16 1 1 0 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x2-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x2 mobilenetv2_ndp_batch_16 1 1 0 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x2-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x2 vgg16_ndp_batch_16 1 1 0 > vgg16_ndp_batch_16-V100_downscaled_HBM2_PCI6x2-1-nosync


#Link 1.5x
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1.5 mobilenetv2_ndp_batch_64 1 1 0 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1.5-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1.5 vgg16_ndp_batch_32 1 1 0 > vgg16_ndp_batch_32-V100_downscaled_HBM2_PCI6x1.5-1-nosync


#Link 2x
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x2 mobilenetv2_ndp_batch_64 1 1 0 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x2-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x2 vgg16_ndp_batch_32 1 1 0 > vgg16_ndp_batch_32-V100_downscaled_HBM2_PCI6x2-1-nosync



# batch 16, 4 GPU, sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet18_ndp_batch_16 4 4 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet50_ndp_batch_16 4 4 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio mobilenetv2_ndp_batch_16 4 4 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-4-sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio vgg16_ndp_batch_16 4 4 1 > vgg16_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-4-sync
#
## batch 16, 4 GPU, no sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet18_ndp_batch_16 4 4 0 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet50_ndp_batch_16 4 4 0 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio mobilenetv2_ndp_batch_16 4 4 0 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-4-nosync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio vgg16_ndp_batch_16 4 4 0 > vgg16_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-4-nosync
#
## batch 16, 2 GPU, sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet18_ndp_batch_16 2 2 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-2-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet50_ndp_batch_16 2 2 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-2-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio mobilenetv2_ndp_batch_16 2 2 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-2-sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio vgg16_ndp_batch_16 2 2 1 > vgg16_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-2-sync
#
## batch 16, 2 GPU, no sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet18_ndp_batch_16 2 2 0 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-2-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet50_ndp_batch_16 2 2 0 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-2-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio mobilenetv2_ndp_batch_16 2 2 0 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-2-nosync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio vgg16_ndp_batch_16 2 2 0 > vgg16_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-2-nosync
#
## batch 16, 1 GPU, no sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet18_ndp_batch_16 1 1 0 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet50_ndp_batch_16 1 1 0 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio mobilenetv2_ndp_batch_16 1 1 0 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-1-nosync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio vgg16_ndp_batch_16 1 1 0 > vgg16_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_write_prio-1-nosync
#
#
#
#
## batch 64, 4 GPU, sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet18_ndp_batch_64 4 4 1 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet50_ndp_batch_64 4 4 1 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-4-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio mobilenetv2_ndp_batch_64 4 4 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-4-sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio vgg16_ndp_batch_32 4 4 1 > vgg16_ndp_batch_32-V100_downscaled_HBM2_PCI6x1_write_prio-4-sync
#
## batch 64, 4 GPU, no sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet18_ndp_batch_64 4 4 0 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet50_ndp_batch_64 4 4 0 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-4-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio mobilenetv2_ndp_batch_64 4 4 0 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-4-nosync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio vgg16_ndp_batch_64 4 4 0 > vgg16_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-4-nosync
#
## batch 64, 2 GPU, sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet18_ndp_batch_64 2 2 1 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-2-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet50_ndp_batch_64 2 2 1 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-2-sync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio mobilenetv2_ndp_batch_64 2 2 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-2-sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio vgg16_ndp_batch_64 2 2 1 > vgg16_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-2-sync
#
## batch 64, 2 GPU, no sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet18_ndp_batch_64 2 2 0 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-2-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet50_ndp_batch_64 2 2 0 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-2-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio mobilenetv2_ndp_batch_64 2 2 0 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-2-nosync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio vgg16_ndp_batch_64 2 2 0 > vgg16_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-2-nosync
#
## batch 64, 1 GPU, no sync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet18_ndp_batch_64 1 1 0 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio resnet50_ndp_batch_64 1 1 0 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-1-nosync
#sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio mobilenetv2_ndp_batch_64 1 1 0 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1_write_prio-1-nosync
##sh generate_simulation_env_fw.sh V100_downscaled_HBM2_PCI6x1_write_prio vgg16_ndp_batch_32 1 1 0 > vgg16_ndp_batch_32-V100_downscaled_HBM2_PCI6x1_write_prio-1-nosync
#
CONFIG=V100_downscaled_HBM2_PCI6x1
MODEL=bert_ndp
BATCH=16
MODEL=$MODEL'_batch_'$BATCH 
GPUS=2
BUFFERS=2
SYNC=0
sh generate_simulation_env_fw_jueon.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC > $MODEL-$CONFIG-$GPUS-nosync
