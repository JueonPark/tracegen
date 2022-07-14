#!/bin/bash
CONFIG=NDPX_baseline_64
MODEL=$1
# MODEL=220714_bert_large_three_encoder_ndpx
# BATCH=3
# MODEL=$MODEL'_batch_'$BATCH
GPUS=1
BUFFERS=1
SYNC=0
sh generate_simulation_env_dl_model.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC > $MODEL-$CONFIG-$GPUS-nosync
