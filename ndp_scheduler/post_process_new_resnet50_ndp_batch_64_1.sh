#!/bin/bash

python post_process.py --model ../new_resnet50_ndp_batch_64 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all
python make_expr_dir.py --model ../new_resnet50_ndp_batch_64 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all
