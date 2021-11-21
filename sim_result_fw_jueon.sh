#!/bin/bash

# batch 16, 4 GPU, sync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 4 4 1 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-sync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 4 4 1 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-sync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 4 4 1 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-sync

# batch 16, 4 GPU, no sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 4 4 0 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 4 4 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 4 4 0 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-nosync

# batch 16, 2 GPU, sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 2 2 1 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 2 2 1 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 2 2 1 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-sync

# batch 16, 2 GPU, no sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 2 2 0 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 2 2 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 2 2 0 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-nosync

# batch 16, 1 GPU, no sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 1 1 0 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-1-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 1 1 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-1-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 1 1 0 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-1-nosync


# batch 64
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_64 4 4 1 1 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_64 4 4 0 1 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_64 2 2 1 1 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-2-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_64 2 2 0 1 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-2-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_64 1 1 0 1 > resnet18_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-1-nosync


# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_64 4 4 1 1 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_64 4 4 0 1 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_64 2 2 1 1 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-2-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_64 2 2 0 1 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-2-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_64 1 1 0 1 > resnet50_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-1-nosync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1_zero_comp_PCI6x1 resnet50_ndp_batch_16 1 1 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_zero_comp
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1_zero_comp_fw resnet50_ndp_batch_16 1 1 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1_zero_comp_fw_ratio
#CONFIG=V100_downscaled_HBM2_PCI6x1_link112_hbm_187
#sh get_sim_result_fw.sh $CONFIG resnet50_ndp_batch_16 1 1 0 1 > resnet50_ndp_batch_16-$CONFIG
#CONFIG=V100_downscaled_HBM2_PCI6x1_link150_hbm_150
#sh get_sim_result_fw.sh $CONFIG resnet50_ndp_batch_16 1 1 0 1 > resnet50_ndp_batch_16-$CONFIG

# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_64 4 4 1 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_64 4 4 0 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-4-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_64 2 2 1 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-2-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_64 2 2 0 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-2-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_64 1 1 0 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1-1-nosync



# # batch 16, 4 GPU, sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 4 4 1 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 4 4 1 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 4 4 1 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-sync

# # batch 16, 4 GPU, no sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 4 4 0 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 4 4 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 4 4 0 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-4-nosync

# # batch 16, 2 GPU, sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 2 2 1 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 2 2 1 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 2 2 1 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-sync

# # batch 16, 2 GPU, no sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 2 2 0 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 2 2 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 2 2 0 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-2-nosync

# # batch 16, 1 GPU, no sync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet18_ndp_batch_16 1 1 0 1 > resnet18_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-1-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 resnet50_ndp_batch_16 1 1 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-1-nosync
# sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1 mobilenetv2_ndp_batch_16 1 1 0 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1-1-nosync


#Link 1.5
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1.5 resnet50_ndp_batch_16 1 1 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1.5-1-nosync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1.5 mobilenetv2_ndp_batch_16 1 1 0 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1.5-1-nosync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1.5 vgg16_ndp_batch_16 1 1 0 1 > vgg16_ndp_batch_16-V100_downscaled_HBM2_PCI6x1.5-1-nosync
#
##Link 2
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x2 resnet50_ndp_batch_16 1 1 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x2-1-nosync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x2 mobilenetv2_ndp_batch_16 1 1 0 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x2-1-nosync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x2 vgg16_ndp_batch_16 1 1 0 1 > vgg16_ndp_batch_16-V100_downscaled_HBM2_PCI6x2-1-nosync
#
#
##Link 1.5
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1.5 mobilenetv2_ndp_batch_64 1 1 0 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1.5-1-nosync
#
##Link 2
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x2 mobilenetv2_ndp_batch_64 1 1 0 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x2-1-nosync

CONFIG=V100_downscaled_HBM2_PCI6x1
MODEL=bert_ndp
BATCH=16
MODEL=$MODEL'_batch_'$BATCH
GPUS=2
BUFFERS=2
SYNC=0
sh get_sim_result_fw_jueon.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC 1 > $MODEL-$CONFIG-$GPUS-nosync


#Link 1.5
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1.5 resnet50_ndp_batch_16 1 1 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x1.5-1-nosync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1.5 mobilenetv2_ndp_batch_16 1 1 0 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x1.5-1-nosync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1.5 vgg16_ndp_batch_16 1 1 0 1 > vgg16_ndp_batch_16-V100_downscaled_HBM2_PCI6x1.5-1-nosync
#
##Link 2
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x2 resnet50_ndp_batch_16 1 1 0 1 > resnet50_ndp_batch_16-V100_downscaled_HBM2_PCI6x2-1-nosync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x2 mobilenetv2_ndp_batch_16 1 1 0 1 > mobilenetv2_ndp_batch_16-V100_downscaled_HBM2_PCI6x2-1-nosync
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x2 vgg16_ndp_batch_16 1 1 0 1 > vgg16_ndp_batch_16-V100_downscaled_HBM2_PCI6x2-1-nosync
#
#
##Link 1.5
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x1.5 mobilenetv2_ndp_batch_64 1 1 0 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x1.5-1-nosync
#
##Link 2
#sh get_sim_result_fw.sh V100_downscaled_HBM2_PCI6x2 mobilenetv2_ndp_batch_64 1 1 0 1 > mobilenetv2_ndp_batch_64-V100_downscaled_HBM2_PCI6x2-1-nosync
