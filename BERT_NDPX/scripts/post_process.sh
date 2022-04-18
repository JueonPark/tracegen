#!/bin/bash

python post_process.py --model ../new_mobilenetv2_ndp_batch_16 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes all

