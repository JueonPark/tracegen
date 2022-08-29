#!/bin/bash

#python post_process_debug.py --model ../resnet18_ndp_batch_16_write_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw --read_or_write w --sync 1
#python post_process_debug.py --model ../resnet18_ndp_batch_16_write_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw --read_or_write w


#python post_process_debug.py --model ../resnet18_ndp_batch_16_read_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw --read_or_write r --sync 1
#python post_process_debug.py --model ../resnet18_ndp_batch_16_read_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw --read_or_write r


python post_process.py --model ../resnet18_ndp_batch_16_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw --sync 1
python post_process.py --model ../resnet18_ndp_batch_16_debug --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes fw
