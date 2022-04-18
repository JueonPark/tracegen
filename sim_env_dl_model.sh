#!/bin/bash
CONFIG=NDPX_baseline_64
MODEL=bert_ndp_cost_model
BATCH=16
MODEL=$MODEL'_batch_'$BATCH
GPUS=1
BUFFERS=1
SYNC=0
sh generate_simulation_env_dl_model.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC > $MODEL-$CONFIG-$GPUS-nosync
