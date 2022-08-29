#!/bin/bash

python post_process.py --model ../new_resnet18_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes all
python make_expr_dir.py --model ../new_resnet18_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes all
