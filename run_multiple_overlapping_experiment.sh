#!/bin/bash

# gets one input - model
MODEL=$1

# make_ndpx_sim_dir.py generates experiment directory.
python multiple_overlap_scheduler/make_ndpx_sim_dir.py --model $MODEL

# make_ndpx_sim_env.sh generates simulation environment.
sh multiple_overlap_scheduler/make_ndpx_sim_env.sh $MODEL

# get_ndpx_sim_result.sh executes the experiment.
sh multiple_overlap_scheduler/get_ndpx_sim_result.sh $MODEL
