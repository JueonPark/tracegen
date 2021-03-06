#!/bin/bash
CONFIG=$1
TARGET_MODEL=$2
GPUS=$3
BUFFERS=$4
SYNC=$5
DEVICE_SETTING=GPU_${GPUS}_Buffer_${BUFFERS}
SIMULATOR_DIR=/home/hhk971/cxl_simulator/cxl-simulator
SIMULATOR_BINARY_DIR=$SIMULATOR_DIR/multi_gpu_simulator/gpu-simulator/bin/release
CONFIG_DIR=/home/shared/CXL_memory_buffer/ASPLOS_simulator/cxl-simulator/sim_configs/$DEVICE_SETTING/$CONFIG
TARGET_TRACE_DIR=`pwd`/traces/$TARGET_MODEL
PARSED_TRACES=$TARGET_TRACE_DIR/packet_32_buffer_${BUFFERS}_gpu_${GPUS}_sync_${SYNC}_simd_8_bw
TARGET_RESULT_DIR=`pwd`"/results/"$DEVICE_SETTING"/"$TARGET_MODEL"/"$CONFIG
if [ ${SYNC} -eq 1 ]; then
    TARGET_RESULT_DIR=$TARGET_RESULT_DIR"/"sync
else
    TARGET_RESULT_DIR=$TARGET_RESULT_DIR"/"nosync
fi
TARGET_RESULT_FORWARD=$TARGET_RESULT_DIR/forward
TARGET_RESULT_BACKWARD=$TARGET_RESULT_DIR/backward
#make directories
mkdir -p $TARGET_RESULT_DIR
mkdir -p $TARGET_RESULT_BACKWARD

PARSED_TRACES=$TARGET_TRACE_DIR/packet_32_buffer_${BUFFERS}_gpu_${GPUS}_sync_${SYNC}_simd_8_bw
cat $TARGET_TRACE_DIR/traces_bw/kernelslist.g.bw | while read line 
do
    if [[ $line == "kernel"* ]]; then
        KERNEL_NUMBER=$(echo $line | sed -e 's/[^0-9]/ /g' -e 's/^ *//g' -e 's/ *$//g')
        RESULT_DIR=$TARGET_RESULT_BACKWARD/$KERNEL_NUMBER
        mkdir -p $RESULT_DIR/output
        echo -e "#!/bin/bash\
                \ncp -r ${SIMULATOR_DIR}/multi_gpu_simulator/gpu-simulator/gpgpu-sim/lib/gcc-7.3.0/cuda-10010/release/ .\
                \ncp ${SIMULATOR_BINARY_DIR}/accel-sim.out .\
                \nexport LD_LIBRARY_PATH=\`pwd\`/release:\$LD_LIBRARY_PATH\
                \n./accel-sim.out -trace ${PARSED_TRACES}/kernel-${KERNEL_NUMBER}/ -config ${CONFIG_DIR}/gpgpusim.config -config ${CONFIG_DIR}/trace.config -cxl_config ${CONFIG_DIR}/cxl.config -num_gpus $GPUS -num_cxl_memory_buffers $BUFFERS" > $RESULT_DIR/run.sh 
    fi

done


PARSED_TRACES=$TARGET_TRACE_DIR/packet_32_buffer_${BUFFERS}_gpu_${GPUS}_sync_${SYNC}_simd_8_fw
cat $TARGET_TRACE_DIR/traces_fw/kernelslist.g.fw | while read line 
do
    if [[ $line == "kernel"* ]]; then
        KERNEL_NUMBER=$(echo $line | sed -e 's/[^0-9]/ /g' -e 's/^ *//g' -e 's/ *$//g')
        RESULT_DIR=$TARGET_RESULT_FORWARD/$KERNEL_NUMBER
        mkdir -p $RESULT_DIR/output
        echo -e "#!/bin/bash\
                \ncp -r ${SIMULATOR_DIR}/multi_gpu_simulator/gpu-simulator/gpgpu-sim/lib/gcc-7.3.0/cuda-10010/release/ .\
                \ncp ${SIMULATOR_BINARY_DIR}/accel-sim.out .\
                \nexport LD_LIBRARY_PATH=\`pwd\`/release:\$LD_LIBRARY_PATH\
                \n./accel-sim.out -trace ${PARSED_TRACES}/kernel-${KERNEL_NUMBER}/ -config ${CONFIG_DIR}/gpgpusim.config -config ${CONFIG_DIR}/trace.config -cxl_config ${CONFIG_DIR}/cxl.config -num_gpus $GPUS -num_cxl_memory_buffers $BUFFERS" > $RESULT_DIR/run.sh 
    fi

done


