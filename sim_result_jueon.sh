#!/bin/bash

CONFIG=V100_downscaled_HBM2_PCI6x1
MODEL=simple_graph_ndpx
GPUS=1
BUFFERS=1
SYNC=0
RESUBMIT=1
sh get_sim_result_jueon.sh $CONFIG $MODEL $GPUS $BUFFERS $SYNC $RESUBMIT > $MODEL-$CONFIG-$GPUS-nosync
