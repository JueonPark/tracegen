#!/bin/bash
CONFIG=NDPX_baseline_64
MODEL=220420_bert_large_cost_model
BATCH=4
MODEL=$MODEL'_batch_'$BATCH
GPUS=1
BUFFERS=1
SYNC=0
RESUBMIT=1
sh get_sim_result_dl_model.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC $RESUBMIT > $MODEL-$CONFIG-$GPUS-nosync
