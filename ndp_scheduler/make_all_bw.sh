#!/bin/bash

#python make_expr_dir.py --model mobilenetv2_ndp_batch_64 --packet-size 32 --gpu 2 --buffer 2 --sync 1 --simd 8 --passes bw
#python make_expr_dir.py --model mobilenetv2_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 4 --sync 1 --simd 8 --passes bw

#python make_expr_dir.py --model mobilenetv2_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --sync 1 --simd 8 --passes bw
#python make_expr_dir.py --model mobilenetv2_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 4 --sync 1 --simd 8 --passes bw

#python make_expr_dir.py --model resnet18_ndp_batch_64 --packet-size 32 --gpu 2 --buffer 2 --sync 1 --simd 8 --passes bw
#python make_expr_dir.py --model resnet18_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 4 --sync 1 --simd 8 --passes bw

#python make_expr_dir.py --model resnet18_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --sync 1 --simd 8 --passes bw
#python make_expr_dir.py --model resnet18_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 4 --sync 1 --simd 8 --passes bw

python make_expr_dir.py --model ../resnet50_ndp_batch_64 --packet-size 32 --gpu 2 --buffer 2 --sync 1 --simd 8 --passes bw
python make_expr_dir.py --model ../resnet50_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 4 --sync 1 --simd 8 --passes bw

python make_expr_dir.py --model ../resnet50_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --sync 1 --simd 8 --passes bw
python make_expr_dir.py --model ../resnet50_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 4 --sync 1 --simd 8 --passes bw
