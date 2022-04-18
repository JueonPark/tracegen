#!/bin/bash
TRACE_PATH=$1

find $TRACE_PATH/packet*  -name "*fw_bert_output_ph1.traceg" -type f | while read trace_file;
do
    ADDR=`grep "SET_FILTER" $trace_file | cut -d" " -f2 | head -n 1`
    DIR_NAME=`dirname $trace_file` 
    LAST_LINE=`grep -n $ADDR $DIR_NAME/kernel* | tail -n1`
    LINE_NO=`echo $LAST_LINE | cut -d: -f2`
    KERNEL=`echo $LAST_LINE | cut -d: -f1`
    sed -i "1,$((LINE_NO -1))s/0x1007/0x7/" $KERNEL
done

