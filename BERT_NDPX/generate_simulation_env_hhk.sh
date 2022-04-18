#!/bin/bash
CONFIG=$1
TARGET_MODEL=$2
FORWARD_END=$3

SIMULATOR_DIR=/home/hhk971/cxl_simulator/cxl-simulator/multi_gpu_simulator/
SIMULATOR_BINARY_DIR=$SIMULATOR_DIR/gpu-simulator/bin/release
CONFIG_DIR=/home/shared/CXL_memory_buffer/ASPLOS_simulator/cxl-simulator/sim_configs/GPU_1_Buffer_1/$CONFIG
TARGET_TRACE_DIR=`pwd`/traces/$TARGET_MODEL
PARSED_TRACES=$TARGET_TRACE_DIR/parsed
TARGET_RESULT_DIR=`pwd`"/results/"$TARGET_MODEL"/"$CONFIG
TARGET_RESULT_FORWARD=$TARGET_RESULT_DIR/forward
TARGET_RESULT_BACKWARD=$TARGET_RESULT_DIR/backward
#make directories
mkdir -p $TARGET_RESULT_DIR
mkdir $TARGET_RESULT_FORWARD
mkdir $TARGET_RESULT_BACKWARD

cat $TARGET_TRACE_DIR/kernelslist.g.fw | while read line 
do
    if [[ $line == "kernel"* ]]; then
        KERNEL_NUMBER=$(echo $line | sed -e 's/[^0-9]/ /g' -e 's/^ *//g' -e 's/ *$//g')
#        RESULT_DIR=$TARGET_RESULT_BACKWARD/$KERNEL_NUMBER 
        RESULT_DIR=$TARGET_RESULT_FORWARD/$KERNEL_NUMBER
        mkdir $RESULT_DIR
        echo -e "#!/bin/bash\
                \ncp -r ${SIMULATOR_DIR}/gpu-simulator/gpgpu-sim/lib/gcc-7.3.0/cuda-10010/release/ $RESULT_DIR/\
                \ncp ${SIMULATOR_BINARY_DIR}/accel-sim.out .\
                \nexport LD_LIBRARY_PATH=\`pwd\`/release:\$LD_LIBRARY_PATH\
                \nmkdir output
                \n./accel-sim.out -trace ${PARSED_TRACES}/${KERNEL_NUMBER}/ -config ${CONFIG_DIR}/gpgpusim.config -config ${CONFIG_DIR}/trace.config" -cxl_config ${CONFIG_DIR}/cxl.config -num_gpus 1 -num_cxl_memory_buffers 1  > $RESULT_DIR/run.sh 
    fi

done

cat $TARGET_TRACE_DIR/kernelslist.g.bw | while read line 
do
    if [[ $line == "kernel"* ]]; then
        KERNEL_NUMBER=$(echo $line | sed -e 's/[^0-9]/ /g' -e 's/^ *//g' -e 's/ *$//g')
        RESULT_DIR=$TARGET_RESULT_BACKWARD/$KERNEL_NUMBER 
       # RESULT_DIR=$TARGET_RESULT_FORWARD/$KERNEL_NUMBER
        mkdir $RESULT_DIR
        echo -e "#!/bin/bash\
                \ncp -r ${SIMULATOR_DIR}/gpu-simulator/gpgpu-sim/lib/gcc-7.3.0/cuda-10010/release/ $RESULT_DIR/\
                \ncp ${SIMULATOR_BINARY_DIR}/accel-sim.out .\
                \nexport LD_LIBRARY_PATH=\`pwd\`/release:\$LD_LIBRARY_PATH\
                \nmkdir output
                \n./accel-sim.out -trace ${PARSED_TRACES}/${KERNEL_NUMBER}/ -config ${CONFIG_DIR}/gpgpusim.config -config ${CONFIG_DIR}/trace.config" -cxl_config ${CONFIG_DIR}/cxl.config -num_gpus 1 -num_cxl_memory_buffers 1  > $RESULT_DIR/run.sh 
    fi

done
