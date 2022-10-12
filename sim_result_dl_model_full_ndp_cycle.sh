#!/bin/bash
CONFIG=NDPX_baseline_64
MODEL=220509_bert_large_cost_model
BATCH=2
MODEL=$MODEL'_batch_'$BATCH
GPUS=1
BUFFERS=1
SYNC=0
RESUBMIT=1
sh get_sim_result_dl_model_full_ndp_cycle.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC $RESUBMIT > $MODEL-$CONFIG-$GPUS-nosync
