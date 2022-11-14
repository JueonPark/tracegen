#!/bin/bash

# gets one input - model
MODEL=$1

# trace_scheduler.py generates kernelslist.g file to the given model path.
# kernelslist.g consists of multiple items of scheduling information:
#   // Thunk: fusion.393
#   // Kernel Name: fusion_393
#   # NO_CXL
#   _NDP_custom-call.128_0_NdpEwiseFused$fusion.393.traceg
#   kernel-8352.traceg
python scheduler/trace_scheduler.py -m $MODEL

# trace_postprocessor.py does postprocessing on GPU traces.
# for data that goes to NDPX, the postprocessor changes the address to the corresponding NDPX address.
python scheduler/trace_postprocessor.py --model $MODEL --packet-size 32 --gpu 1 --buffer 1 --simd 8

# trace_make_expr_dir.py generates the experiment directory.
# each experiment directory consists of:
# kernel_directory
# - GPU trace
# - NDPX trace (if exist)
# - kernelslist.g - this file shows the sequence of traces to run.
# kernelslist.g - this file shows which GPU trace to run.
python scheduler/trace_make_expr_dir.py --model $MODEL --packet-size 32 --gpu 1 --buffer 1 --simd 8

# sim_env_jueon.sh generates the simulation enviornment to run.
# it generates the result directory on results/
sh sim_env_jueon.sh $MODEL

# this script runs the simulation.
sh sim_result_jueon.sh $MODEL
