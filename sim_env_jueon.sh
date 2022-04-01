#!/bin/bash
CONFIG=NDPX_baseline_64
MODEL=simple_graph_ndpx  # model is the directory of traces
GPUS=1
BUFFERS=1
SYNC=0
sh generate_simulation_env_jueon.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC > $MODEL-$CONFIG-$GPUS-nosync
