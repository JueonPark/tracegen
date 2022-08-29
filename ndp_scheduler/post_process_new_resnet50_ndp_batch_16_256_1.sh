#!/bin/bash

python post_process.py --model ../resnet50_ndp_batch_16_opt --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all
python make_expr_dir.py --model ../resnet50_ndp_batch_16_opt --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all

python post_process.py --model ../resnet18_ndp_batch_16_opt --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all
python make_expr_dir.py --model ../resnet18_ndp_batch_16_opt --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all

python post_process.py --model ../mobilenetv2_ndp_batch_16_opt --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all
python make_expr_dir.py --model ../mobilenetv2_ndp_batch_16_opt --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all

python post_process.py --model ../vgg16_ndp_batch_16_opt --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all
python make_expr_dir.py --model ../vgg16_ndp_batch_16_opt --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all
