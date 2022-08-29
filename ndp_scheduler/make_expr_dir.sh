#!/bin/bash

python make_expr_dir.py --model ../new_resnet18_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw
#python make_expr_dir.py --model ../resnet18_small --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_16_128 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_16_128 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../resnet18_ndp_batch_16_128 --packet-size 128 --gpu 4 --buffer 4 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_16_128 --packet-size 128 --gpu 4 --buffer 4 --simd 8 --passes all --sync 1



#python make_expr_dir.py --model ../resnet18_ndp_batch_16 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1

#python make_expr_dir.py --model ../resnet50_ndp_batch_16 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet50_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet50_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../resnet50_ndp_batch_64 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet50_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet50_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1

#python make_expr_dir.py --model ../mobilenetv2_ndp_batch_16 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../mobilenetv2_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python make_expr_dir.py --model ../mobilenetv2_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../mobilenetv2_ndp_batch_64 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../mobilenetv2_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python make_expr_dir.py --model ../mobilenetv2_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1

#python make_expr_dir.py --model ../vgg16_ndp_batch_16 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../vgg16_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python make_expr_dir.py --model ../vgg16_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../vgg16_ndp_batch_32 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../vgg16_ndp_batch_32 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python make_expr_dir.py --model ../vgg16_ndp_batch_32 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1





#python make_expr_dir.py --model ../bert_ndp_batch_16 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes fw
#python make_expr_dir.py --model ../bert_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw
#python make_expr_dir.py --model ../bert_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes bw 
#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 32 --gpu 16 --buffer 16 --simd 8 --passes bw
#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 32 --gpu 16 --buffer 16 --simd 8 --passes bw --sync 1

#python make_expr_dir.py --model ../resnet18_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../resnet50_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet50_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../resnet50_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet50_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1

#python make_expr_dir.py --model ../mobilenetv2_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../mobilenetv2_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../mobilenetv2_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../mobilenetv2_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../vgg16_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../vgg16_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../vgg16_ndp_batch_32 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../vgg16_ndp_batch_32 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1
#python make_expr_dir.py --model ../resnet18_ndp_batch_16_large_gpu --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_16_large_gpu --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1



#python make_expr_dir.py --model ../resnet18_ndp_batch_64_large_gpu --packet-size 32 --gpu 8 --buffer 8 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_64_large_gpu --packet-size 32 --gpu 8 --buffer 8 --simd 8 --passes all --sync 1

#python make_expr_dir.py --model ../resnet18_ndp_batch_64_large_gpu --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python make_expr_dir.py --model ../resnet18_ndp_batch_64_large_gpu --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1

#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes bw
#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes bw --sync 1
#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw
#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw --sync 1
#python make_expr_dir.py --model ../resnet18_ndp_batch_64 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes bw


#python make_expr_dir.py --model ../resnet18_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes bw
#python make_expr_dir.py --model ../resnet18_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes bw --sync 1
#python make_expr_dir.py --model ../resnet18_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw
#python make_expr_dir.py --model ../resnet18_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw --sync 1
#python make_expr_dir.py --model ../resnet18_ndp_batch_16 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes bw



#python make_expr_dir.py --model ../vgg16_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes bw
#python make_expr_dir.py --model ../vgg16_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes bw --sync 1
#python make_expr_dir.py --model ../vgg16_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw
#python make_expr_dir.py --model ../vgg16_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw --sync 1

#python make_expr_dir.py --model ../vgg16_ndp_batch_32 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes bw
#python make_expr_dir.py --model ../vgg16_ndp_batch_32 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes bw --sync 1
#python make_expr_dir.py --model ../vgg16_ndp_batch_32 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw
#python make_expr_dir.py --model ../vgg16_ndp_batch_32 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw --sync 1


#python make_expr_dir.py --model ../resnet18_ndp_batch_16_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw
#python make_expr_dir.py --model ../resnet18_ndp_batch_16_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw --sync 1
#python make_expr_dir.py --model ../resnet18_ndp_batch_16_write_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw
#python make_expr_dir.py --model ../resnet18_ndp_batch_16_write_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw --sync 1
#python make_expr_dir.py --model ../resnet18_ndp_batch_16_read_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw
#python make_expr_dir.py --model ../resnet18_ndp_batch_16_read_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw --sync 1
