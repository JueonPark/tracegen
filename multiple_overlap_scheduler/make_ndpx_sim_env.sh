#!/bin/bash
CONFIG=NDPX_baseline_64
GPUS=1
BUFFERS=1
SIMULATOR_DIR=/home/jueonpark/cxl-simulator
SIMULATOR_BINARY_DIR=$SIMULATOR_DIR/multi_gpu_simulator/gpu-simulator/bin/release
DEVICE_SETTING=GPU_${GPUS}_Buffer_${BUFFERS}
CONFIG_DIR=/home/shared/CXL_memory_buffer/ASPLOS_simulator/cxl-simulator/sim_configs/$DEVICE_SETTING/$CONFIG
TRACE_DIR=./traces/$1/exp_trace_dir
RESULT_DIR=./results/GPU_${GPUS}_Buffer_${BUFFERS}/$1

#make directories
mkdir -p $RESULT_DIR

for ITEM in `ls $TRACE_DIR`;
do
  RESULT_ITEM_DIR=$RESULT_DIR/$ITEM
  mkdir -p $RESULT_ITEM_DIR/output
  echo -e "#!/bin/bash\
          \ncp -r ${SIMULATOR_DIR}/multi_gpu_simulator/gpu-simulator/gpgpu-sim/lib/gcc-7.3.0/cuda-10010/release/ .\
          \ncp ${SIMULATOR_BINARY_DIR}/accel-sim.out .\
          \nexport LD_LIBRARY_PATH=\`pwd\`/release:\$LD_LIBRARY_PATH\
          \n./accel-sim.out -trace ${TRACE_DIR}/${ITEM}/ -config ${CONFIG_DIR}/gpgpusim.config -config ${CONFIG_DIR}/trace.config -cxl_config ${CONFIG_DIR}/cxl.config -num_gpus $GPUS -num_cxl_memory_buffers $BUFFERS" > $RESULT_ITEM_DIR/run.sh 
done