#!/bin/bash

# gets one input - model
MODEL=$1

python BERT_NDPX/scripts/dependency_scheduler.py -m $MODEL

python preprocess_traces.py -m $MODEL

python BERT_NDPX/scripts/post_process.py --model $MODEL --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all

python BERT_NDPX/scripts/make_expr_dir_hhk.py --model $MODEL --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all

sh sim_env_dl_model.sh $MODEL

sh sim_result_dl_model.sh $MODEL
