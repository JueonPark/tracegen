#!/bin/bash
CONFIG=NDPX_baseline_64
# MODEL is the directory of traces
MODEL=ndpx_elementwise_524288
GPUS=1
BUFFERS=1
SYNC=0
RESUBMIT=1
sh get_sim_result_jueon.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC $RESUBMIT > $MODEL-$CONFIG-$GPUS-nosync
