#!/bin/bash
source generate_trace_preprocessing.bash_profile
rm -rf hlo_graph traces output
./make_ndpx_trace_dir.sh
LD_PRELOAD=$TRACER_TOOL TF_DUMP_GRAPH_PREFIX=$NDP_TRACE_DIR TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo_graph --xla_gpu_cuda_data_dir=/home/jueonpark/cuda-10.1" python $TEST_PROGRAM
