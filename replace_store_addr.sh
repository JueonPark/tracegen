

b2g2=packet_32_buffer_2_gpu_2_sync_0_simd_8
b4g4=packet_32_buffer_4_gpu_4_sync_0_simd_8

python replace_store_addr.py --filename ./traces/bert_ndp_batch_16/$b2g2'_fw'/kernel-8156/GPU_0/kernel-8156.traceg
python replace_store_addr.py --filename ./traces/bert_ndp_batch_16/$b4g4'_fw'/kernel-8156/GPU_0/kernel-8156.traceg

python replace_store_addr.py --filename ./traces/bert_ndp_batch_16/$b2g2'_fw'/kernel-8160/GPU_0/kernel-8160.traceg
python replace_store_addr.py --filename ./traces/bert_ndp_batch_16/$b4g4'_fw'/kernel-8160/GPU_0/kernel-8160.traceg

python replace_store_addr.py --filename ./traces/bert_ndp_batch_16/$b2g2'_fw'/kernel-8178/GPU_0/kernel-8178.traceg
python replace_store_addr.py --filename ./traces/bert_ndp_batch_16/$b4g4'_fw'/kernel-8178/GPU_0/kernel-8178.traceg

python replace_store_addr.py --filename ./traces/bert_ndp_batch_16/$b2g2'_fw'/kernel-8182/GPU_0/kernel-8182.traceg
python replace_store_addr.py --filename ./traces/bert_ndp_batch_16/$b4g4'_fw'/kernel-8182/GPU_0/kernel-8182.traceg

