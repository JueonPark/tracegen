#!/bin/bash

python post_process.py --model ../new_resnet18_ndp_batch_16_256 --packet-size 256 --gpu 1 --buffer 1 --simd 8 --passes all
python make_expr_dir.py --model ../new_resnet18_ndp_batch_16_256 --packet-size 256 --gpu 1 --buffer 1 --simd 8 --passes all
