#!/bin/bash

python post_process.py --model ../new_mobilenetv2_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes all
python make_expr_dir.py --model ../new_mobilenetv2_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes all
