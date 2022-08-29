#!/bin/bash

python post_process.py --model ../new_resnet18_ndp_batch_16 --packet-size 32 --gpu 8 --buffer 8 --simd 8 --passes all
python make_expr_dir.py --model ../new_resnet18_ndp_batch_16 --packet-size 32 --gpu 8 --buffer 8 --simd 8 --passes all
