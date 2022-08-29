#!/bin/bash

python post_process_align.py --model ../resnet18_ndp_batch_16 --packet-size 128 --gpu 4 --buffer 4 --simd 8
python post_process_align.py --model ../resnet18_ndp_batch_16 --packet-size 128 --gpu 4 --buffer 4 --simd 8 --sync 1

#python post_process_align.py --model ../resnet18_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8
#python post_process_align.py --model ../resnet18_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --sync 1
#python post_process_align.py --model ../resnet18_ndp_batch_64 --packet-size 128 --gpu 1 --buffer 1 --simd 8

#python post_process_align.py --model ../resnet50_ndp_batch_16 --packet-size 128 --gpu 1 --buffer 1 --simd 8
#python post_process_align.py --model ../resnet50_ndp_batch_64 --packet-size 128 --gpu 1 --buffer 1 --simd 8

#python post_process_align.py --model ../mobilenetv2_ndp_batch_16 --packet-size 128 --gpu 1 --buffer 1 --simd 8
#python post_process_align.py --model ../mobilenetv2_ndp_batch_64 --packet-size 128 --gpu 1 --buffer 1 --simd 8

#python post_process_align.py --model ../vgg16_ndp_batch_16 --packet-size 128 --gpu 1 --buffer 1 --simd 8
#python post_process_align.py --model ../vgg16_ndp_batch_32 --packet-size 128 --gpu 1 --buffer 1 --simd 8
