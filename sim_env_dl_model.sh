#!/bin/bash
CONFIG=NDPX_baseline_64
MODEL=220420_bert_large_cost_model
BATCH=4
MODEL=$MODEL'_batch_'$BATCH
GPUS=1
BUFFERS=1
SYNC=0
sh generate_simulation_env_dl_model.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC > $MODEL-$CONFIG-$GPUS-nosync
