#!/bin/bash

# gets one input - model
MODEL=$1

python scheduler/trace_scheduler.py -m $MODEL

python scheduler/trace_postprocessor.py --model $MODEL --packet-size 32 --gpu 1 --buffer 1 --simd 8

python scheduler/trace_make_expr_dir.py --model $MODEL --packet-size 32 --gpu 1 --buffer 1 --simd 8

sh sim_env_jueon.sh $MODEL

sh sim_result_jueon.sh $MODEL
