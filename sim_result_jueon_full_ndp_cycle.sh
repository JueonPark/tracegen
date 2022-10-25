#!/bin/bash
CONFIG=NDPX_baseline_64
MODEL=$1
GPUS=1
BUFFERS=1
SYNC=0
RESUBMIT=1
sh get_sim_result_jueon_full_ndp_cycle.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC $RESUBMIT > $MODEL-$CONFIG-$GPUS-nosync