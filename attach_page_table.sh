

b2g2=packet_32_buffer_2_gpu_2_sync_0_simd_8
b4g4=packet_32_buffer_4_gpu_4_sync_0_simd_8

python attach_page_table.py --trace ./traces/bert_ndp_batch_16/$b2g2'_fw'/kernel-8157/GPU_0/kernel-8157.traceg --page_table ./traces/bert_ndp_batch_16/$b2g2/page_table_header_custom-call.74.traceg 
python attach_page_table.py --trace ./traces/bert_ndp_batch_16/$b4g4'_fw'/kernel-8157/GPU_0/kernel-8157.traceg  --page_table ./traces/bert_ndp_batch_16/$b4g4/page_table_header_custom-call.74.traceg

python attach_page_table.py --trace ./traces/bert_ndp_batch_16/$b2g2'_fw'/kernel-8161/GPU_0/kernel-8161.traceg  --page_table ./traces/bert_ndp_batch_16/$b2g2/page_table_header_custom-call.75.traceg
python attach_page_table.py --trace ./traces/bert_ndp_batch_16/$b4g4'_fw'/kernel-8161/GPU_0/kernel-8161.traceg  --page_table ./traces/bert_ndp_batch_16/$b4g4/page_table_header_custom-call.75.traceg

python attach_page_table.py --trace ./traces/bert_ndp_batch_16/$b2g2'_fw'/kernel-8179/GPU_0/kernel-8179.traceg  --page_table ./traces/bert_ndp_batch_16/$b2g2/page_table_header_custom-call.76.traceg
python attach_page_table.py --trace ./traces/bert_ndp_batch_16/$b4g4'_fw'/kernel-8179/GPU_0/kernel-8179.traceg  --page_table ./traces/bert_ndp_batch_16/$b4g4/page_table_header_custom-call.76.traceg

