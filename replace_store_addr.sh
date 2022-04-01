#!/bin/bash

b1g1=packet_32_buffer_1_gpu_1_sync_0_simd_8
b2g2=packet_32_buffer_2_gpu_2_sync_0_simd_8
b4g4=packet_32_buffer_4_gpu_4_sync_0_simd_8

# python replace_store_addr.py --filename /home/jueonpark/isca/bert_ndp_batch_16/$b1g1'_fw'/kernel-8136/GPU_0/kernel-8136.traceg
# python replace_store_addr.py --filename /home/jueonpark/isca/bert_ndp_batch_16/$b2g2'_fw'/kernel-8136/GPU_0/kernel-8136.traceg
# python replace_store_addr.py --filename /home/jueonpark/isca/bert_ndp_batch_16/$b4g4'_fw'/kernel-8136/GPU_0/kernel-8136.traceg

python replace_store_addr.py --filename ./traces/kernel-$1/GPU_0/kernel-$1.traceg
