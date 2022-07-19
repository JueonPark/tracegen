#!/bin/bash
shopt -s extglob
# setup environment
source setup_environment_jueon.sh

# remove checkpoints
rm /home/shared/CXL_memory_buffer/tensorflow_models/datasets/*ckpt* && rm /home/shared/CXL_memory_buffer/tensorflow_models/datasets/pretrained/*ckpt*
# remove files
rm xla_hlo/!("offline_execution_result.csv")
rm xla_hlo/*/*

# simulation
export TRACER_TOOL=/home/jueonpark/cxl-simulator/multi_gpu_simulator/util/tracer_nvbit/tracer_tool/tracer_tool.so
export POST_PROCESSING=/home/jueonpark/cxl-simulator/multi_gpu_simulator/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing
# caution!
# bert large, batch 2, one encoder layer: 7876
# bert large, batch 4, one encoder layer: 8185
# bert large, batch 3, three encoder layer: 8351
export DYNAMIC_KERNEL_LIMIT_START=8351
export DYNAMIC_KERNEL_LIMIT_END=9999999

# batch size to run
BATCH=3

# additional runtime environment variables for tensorflow
# export TF_CPP_MIN_VLOG_LEVEL=1
# export ENABLE_CONSOLE=true

# execution options:
# $1:
# - da for ndpx device assignment
# - pm for pattern matching only
# - daonly for ndpx device assignment only
# $2: trace generation
# - keyword "trace" given
# $3: xla_ndpx_use_offline_result
# - 0 for using GPU results
# - 1 for using SIM results
if [ $1 = "da" ]
then
  if [ $# = 1 ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_optimizer=true  --xla_ndpx_use_device_assignment=true --xla_dump_to=./xla_hlo "
    python run_pretraining.py --input_files /home/shared/CXL_memory_buffer/tensorflow_models/model2.4/models/official/nlp/data/pretrained.tfrecord --bert_config_file=$BERT_DIR/bert_config.json --model_dir=$MODEL_DIR --train_batch_size=$BATCH --max_predictions_per_seq=76 --max_seq_length=512 --optimizer_type=adam --num_steps_per_epoch=1 --num_train_epochs=1 --enable_xla --dtype fp16 1> output 2> error
  elif [ $2 = "trace" ]
  then
    # used when getting traces
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_optimizer=true --xla_ndpx_use_device_assignment=true --xla_dump_to=./xla_hlo "
    LD_PRELOAD=$TRACER_TOOL python run_pretraining.py --input_files /home/shared/CXL_memory_buffer/tensorflow_models/model2.4/models/official/nlp/data/pretrained.tfrecord --bert_config_file=$BERT_DIR/bert_config.json --model_dir=$MODEL_DIR --train_batch_size=$BATCH --max_predictions_per_seq=76 --max_seq_length=512 --optimizer_type=adam --num_steps_per_epoch=1 --num_train_epochs=1 --enable_xla --dtype fp16 1> output 2> error
    $POST_PROCESSING ./traces/kernelslist
  fi
elif [ $1 = "pm" ]
then
  if [ $2 = "trace" ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_optimizer=true --xla_ndpx_use_device_assignment=false --xla_dump_to=./xla_hlo "
    LD_PRELOAD=$TRACER_TOOL python run_pretraining.py --input_files /home/shared/CXL_memory_buffer/tensorflow_models/model2.4/models/official/nlp/data/pretrained.tfrecord --bert_config_file=$BERT_DIR/bert_config.json --model_dir=$MODEL_DIR --train_batch_size=$BATCH --max_predictions_per_seq=76 --max_seq_length=512 --optimizer_type=adam --num_steps_per_epoch=1 --num_train_epochs=1 --enable_xla --dtype fp16 1> output 2> error
    $POST_PROCESSING ./traces/kernelslist
  else
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_optimizer=true --xla_ndpx_use_device_assignment=false --xla_dump_to=./xla_hlo "
    python run_pretraining.py --input_files /home/shared/CXL_memory_buffer/tensorflow_models/model2.4/models/official/nlp/data/pretrained.tfrecord --bert_config_file=$BERT_DIR/bert_config.json --model_dir=$MODEL_DIR --train_batch_size=$BATCH --max_predictions_per_seq=76 --max_seq_length=512 --optimizer_type=adam --num_steps_per_epoch=1 --num_train_epochs=1 --enable_xla --dtype fp16 1> output 2> error
  fi
elif [ $1 = "daonly" ]
then
  if [ $# = 1 ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_ndp_bert_pattern=false --xla_gpu_use_ndp_batchnorm=false --xla_gpu_use_ndp_optimizer=false --xla_ndpx_use_device_assignment=true --xla_dump_to=./xla_hlo "
    python run_pretraining.py --input_files /home/shared/CXL_memory_buffer/tensorflow_models/model2.4/models/official/nlp/data/pretrained.tfrecord --bert_config_file=$BERT_DIR/bert_config.json --model_dir=$MODEL_DIR --train_batch_size=$BATCH --max_predictions_per_seq=76 --max_seq_length=512 --optimizer_type=adam --num_steps_per_epoch=1 --num_train_epochs=1 --enable_xla --dtype fp16 1> output 2> error
  elif [ $2 = "trace" ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_ndp_bert_pattern=false --xla_gpu_use_ndp_batchnorm=false --xla_gpu_use_ndp_optimizer=false --xla_ndpx_use_device_assignment=true --xla_dump_to=./xla_hlo "
    LD_PRELOAD=$TRACER_TOOL python run_pretraining.py --input_files /home/shared/CXL_memory_buffer/tensorflow_models/model2.4/models/official/nlp/data/pretrained.tfrecord --bert_config_file=$BERT_DIR/bert_config.json --model_dir=$MODEL_DIR --train_batch_size=$BATCH --max_predictions_per_seq=76 --max_seq_length=512 --optimizer_type=adam --num_steps_per_epoch=1 --num_train_epochs=1 --enable_xla --dtype fp16 1> output 2> error
    $POST_PROCESSING ./traces/kernelslist
  fi
elif [ $1 = "vanila" ]
then
  if [ $# = 1 ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./xla_hlo "
    python run_pretraining.py --input_files /home/shared/CXL_memory_buffer/tensorflow_models/model2.4/models/official/nlp/data/pretrained.tfrecord --bert_config_file=$BERT_DIR/bert_config.json --model_dir=$MODEL_DIR --train_batch_size=$BATCH --max_predictions_per_seq=76 --max_seq_length=512 --optimizer_type=adam --num_steps_per_epoch=1 --num_train_epochs=1 --enable_xla --dtype fp16 1> output 2> error
  elif [ $2 = "trace" ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./xla_hlo "
    LD_PRELOAD=$TRACER_TOOL python run_pretraining.py --input_files /home/shared/CXL_memory_buffer/tensorflow_models/model2.4/models/official/nlp/data/pretrained.tfrecord --bert_config_file=$BERT_DIR/bert_config.json --model_dir=$MODEL_DIR --train_batch_size=$BATCH --max_predictions_per_seq=76 --max_seq_length=512 --optimizer_type=adam --num_steps_per_epoch=1 --num_train_epochs=1 --enable_xla --dtype fp16 1> output 2> error
    $POST_PROCESSING ./traces/kernelslist
  fi
fi

sh change_file_to_mlir.sh
