#!/bin/bash
CONFIG=V100_downscaled_HBM2_PCI6x1
MODEL=bert_ndp
BATCH=16
MODEL=$MODEL'_batch_'$BATCH 
GPUS=2
BUFFERS=2
SYNC=0
sh generate_simulation_env_fw_jueon.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC > $MODEL-$CONFIG-$GPUS-nosync
