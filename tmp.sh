config_dir=$1
custom_call=$2
trace=$3

python attach_page_table.py --trace ~/isca/bert_ndp_batch_16/$config_dir/kernel-$trace/GPU_0/kernel-$trace.traceg --page_table ~/isca/bert_ndp_batch_16/$config_dir/kernel-$trace/GPU_0/page_table_header_custom-call.$custom_call.traceg