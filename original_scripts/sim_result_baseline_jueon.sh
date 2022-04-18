#!/bin/bash
CONFIG=NDPX_baseline_64
MODEL=simple_graph_baseline
GPUS=1
BUFFERS=1
SYNC=0
RESUBMIT=1
sh get_sim_result_baseline_jueon.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC $RESUBMIT > $MODEL-$CONFIG-$GPUS-nosync
