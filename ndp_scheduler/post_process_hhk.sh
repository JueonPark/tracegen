#!/bin/bash
#python3 post_process.py --model ../../NDP_ISCA/traces/bert_large_3_encoder_b2 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes fw & 
#python3 post_process.py --model ../../NDP_ISCA/traces/bert_large_3_encoder_b2 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes bw &
#python3 post_process.py --model ../../NDP_ISCA/traces/bert_large_3_encoder_b2 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes fw &
#python3 post_process.py --model ../../NDP_ISCA/traces/bert_large_3_encoder_b2 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw &

python3 post_process.py --model ../../NDP_ISCA/traces/BERT_large_3_b8 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes fw 
python3 post_process.py --model ../../NDP_ISCA/traces/BERT_large_3_b8 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes bw 
python3 post_process.py --model ../../NDP_ISCA/traces/BERT_large_3_b8 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes fw 
python3 post_process.py --model ../../NDP_ISCA/traces/BERT_large_3_b8 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw 

#python3 post_process.py --model ../../NDP_ISCA/traces/bert_large_3_encoder_b8 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes fw &
#python3 post_process.py --model ../../NDP_ISCA/traces/bert_large_3_encoder_b8 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes bw &
#python3 post_process.py --model ../../NDP_ISCA/traces/bert_large_3_encoder_b8 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes fw &
#python3 post_process.py --model ../../NDP_ISCA/traces/bert_large_3_encoder_b8 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw &

#python post_process.py --model ../resnet18_ndp_batch_16_128 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python post_process.py --model ../resnet18_ndp_batch_16_128 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1
#python post_process.py --model ../resnet18_ndp_batch_16_128 --packet-size 128 --gpu 4 --buffer 4 --simd 8 --passes all
#python post_process.py --model ../resnet18_ndp_batch_16_128 --packet-size 128 --gpu 4 --buffer 4 --simd 8 --passes all --sync 1
#python post_process.py --model ../resnet18_ndp_batch_16 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../resnet18_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python post_process.py --model ../resnet18_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1
#python post_process.py --model ../resnet18_ndp_batch_64 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../resnet18_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python post_process.py --model ../resnet18_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1

#python post_process.py --model ../resnet50_ndp_batch_16 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../resnet50_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python post_process.py --model ../resnet50_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1
#python post_process.py --model ../resnet50_ndp_batch_64 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../resnet50_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python post_process.py --model ../resnet50_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1

#python post_process.py --model ../mobilenetv2_ndp_batch_16 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../mobilenetv2_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python post_process.py --model ../mobilenetv2_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1
#python post_process.py --model ../mobilenetv2_ndp_batch_64 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../mobilenetv2_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python post_process.py --model ../mobilenetv2_ndp_batch_64 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1

#python post_process.py --model ../vgg16_ndp_batch_16 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../vgg16_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python post_process.py --model ../vgg16_ndp_batch_16 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1
#python post_process.py --model ../vgg16_ndp_batch_32 --packet-size 128 --gpu 1 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../vgg16_ndp_batch_32 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all
#python post_process.py --model ../vgg16_ndp_batch_32 --packet-size 128 --gpu 2 --buffer 2 --simd 8 --passes all --sync 1

#python post_process.py --model ../bert_ndp_batch_16 --packet-size 32 --gpu 1 --buffer 1 --simd 8 --passes fw
#python post_process.py --model ../bert_ndp_batch_16 --packet-size 32 --gpu 2 --buffer 2 --simd 8 --passes bw
#python post_process.py --model ../bert_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 4 --simd 8 --passes bw

#python post_process.py --model ../resnet18_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../resnet18_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1

#python post_process.py --model ../resnet18_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../resnet18_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1


#python post_process.py --model ../resnet50_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../resnet50_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1

#python post_process.py --model ../resnet50_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../resnet50_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1


#python post_process.py --model ../mobilenetv2_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../mobilenetv2_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1

#python post_process.py --model ../mobilenetv2_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../mobilenetv2_ndp_batch_64 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1


#python post_process.py --model ../vgg16_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../vgg16_ndp_batch_16 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1

#python post_process.py --model ../vgg16_ndp_batch_32 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../vgg16_ndp_batch_32 --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1
#python post_process.py --model ../resnet50_ndp_batch_64_large_gpu --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all
#python post_process.py --model ../resnet50_ndp_batch_64_large_gpu --packet-size 32 --gpu 4 --buffer 1 --simd 8 --passes all --sync 1
