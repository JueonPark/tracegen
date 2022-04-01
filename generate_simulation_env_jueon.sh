#!/bin/bash
CONFIG=$1
TARGET_MODEL=$2
GPUS=$3
BUFFERS=$4
SYNC=$5
DEVICE_SETTING=GPU_${GPUS}_Buffer_${BUFFERS}
SIMULATOR_DIR=/home/jueonpark/cxl-simulator
SIMULATOR_BINARY_DIR=$SIMULATOR_DIR/multi_gpu_simulator/gpu-simulator/bin/release
CONFIG_DIR=/home/shared/CXL_memory_buffer/ASPLOS_simulator/cxl-simulator/sim_configs/$DEVICE_SETTING/$CONFIG
TARGET_TRACE_DIR=`pwd`/traces/$TARGET_MODEL/traces
PARSED_TRACES=$TARGET_TRACE_DIR/packet_32_buffer_${BUFFERS}_gpu_${GPUS}_sync_${SYNC}_simd_8_fw
TARGET_RESULT_DIR=`pwd`"/results/"$DEVICE_SETTING"/"$TARGET_MODEL"/"$CONFIG
if [ ${SYNC} -eq 1 ]; then
    TARGET_RESULT_DIR=$TARGET_RESULT_DIR"/"sync
else
    TARGET_RESULT_DIR=$TARGET_RESULT_DIR"/"nosync
fi
#make directories
mkdir -p $TARGET_RESULT_DIR

cat $TARGET_TRACE_DIR/kernelslist.g | while read line 
do
    if [[ $line == "kernel"* ]]; then
        KERNEL_NUMBER=$(echo $line | sed -e 's/[^0-9]/ /g' -e 's/^ *//g' -e 's/ *$//g')
        RESULT_DIR=$TARGET_RESULT_DIR/$KERNEL_NUMBER
        mkdir -p $RESULT_DIR/output
        echo -e "#!/bin/bash\
                \ncp -r ${SIMULATOR_DIR}/multi_gpu_simulator/gpu-simulator/gpgpu-sim/lib/gcc-7.3.0/cuda-10010/release/ .\
                \ncp ${SIMULATOR_BINARY_DIR}/accel-sim.out .\
                \nexport LD_LIBRARY_PATH=\`pwd\`/release:\$LD_LIBRARY_PATH\
                \n./accel-sim.out -trace ${TARGET_TRACE_DIR}/kernel-${KERNEL_NUMBER}/ -config ${CONFIG_DIR}/gpgpusim.config -config ${CONFIG_DIR}/trace.config -cxl_config ${CONFIG_DIR}/cxl.config -num_gpus $GPUS -num_cxl_memory_buffers $BUFFERS" > $RESULT_DIR/run.sh 
    fi

done
