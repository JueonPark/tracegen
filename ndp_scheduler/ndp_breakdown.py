"""
breaks down the gpu kernel
creates a json file that breaks kernel numbers into operation types
애매한 애들은 AMBIGUOUS에 넣어둔다. 얘는 나중에 사람이 손으로 직접
"""
import re
import json
from utils import parse_stats
from utils import parse_hlo

# unconfigured kernel names: 
# - turing_fp16_s884gemm - if from bert: always gemm (linear layer)

LEGENDS = [
    "CONV",
    "POOL",
    "FC",
    "BN & ELEMWISE OP",
    "dxCONV",
    "dwCONV",
    "dxFC",
    "dwFC",
    "AMBIGUOUS",
    "OTHERS"
]

CONV_LIST = [
    "convolve_common_engine",
    "conv2d_c1_k1",
    "first_layer_fwd_kernel"
]

CONV_WGRAD_LIST = [
    # "wgrad2d_c1_k1_nhwc",
    # "first_layer_wgrad",
    # "wgrad2d",
    "wgrad"
]

CONV_DGRAD_LIST = [
    "dgrad"
]

FC_LIST = [
    "cublasGemvTensorStridedBatched"
]

DWFC_LIST = [
    
]

DXFC_LIST = [

]

OTHERS_LIST = [
    "computeOffsetsKernel",
    "ComputeOffsetsParams",
    "computeBOffsetsKernel",
    "scalePackedTensor",
    "select_and_scatter",
    "ComputeBOffsetsParams",
    "genericTranspose",
    "EigenMetaKernel",
    "convertTensor",
    "nhwcToFoldedNhwcKernel",
    "NhwcToNhwcKernel"
]

AMBIGUOUS = [
    "cublasGemvTensorStridedBatched",
    "turing_fp16_s884gemm",
    "turing_fp16_s1688",    # conv or dgrad
]

def is_wgrad(kernel_name):
    return any([wgrad_string in kernel_name for wgrad_string in CONV_WGRAD_LIST])

def is_dgrad(kernel_name):
    return any([dgrad_string in kernel_name for dgrad_string in CONV_DGRAD_LIST])

def is_dwfc(kernel_name):
    return any([dw_string in kernel_name for dw_string in DWFC_LIST])

def is_dxfc(kernel_name):
    return any([dx_string in kernel_name for dx_string in DXFC_LIST])

def is_fc(kernel_name):
    return any([fc_string in kernel_name for fc_string in FC_LIST])

def is_conv(kernel_name):
    return any([conv_string in kernel_name for conv_string in CONV_LIST])

def is_others(kernel_name):
    return any([others_string in kernel_name for others_string in OTHERS_LIST])

def is_xla_kernel(kernel_name):
    # TODO: catch other strings such as convert_32, add, mul_2, ...
    pattern = re.compile('[a-z]+(_[0-9][0-9]*)?$')
    if pattern.match(kernel_name):
        return True
    elif (kernel_name.find("fusion") != -1):
        return True
    else:
        return False
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats-path', type=str, required=True, help="stats.csv file path")
    parser.add_argument('--hlo-path', type=str, required=True, help="xla hlo graph path (text)")
    parser.add_argument('--output-name', type=str, required=False, default="kernel_breakdown.json", help="OUTPUT NAME")
    args = parser.parse_args()

    with open(args.stats_path, 'r') as stats_file:
        parsed_stats = parse_stats(stats_file.read())

    hlo_file = open(args.hlo_path, 'r')
    metadata_table = parse_hlo(hlo_file.read()).hlo_table
    output = {
        "CONV" : [],
        "dxCONV" : [],
        "dwCONV" : [],
        "FC"   : [],
        "dxFC" : [],
        "dwFC" : [],
        "BN & ELEMWISE OP" : [],
        "POOL" : [],
        "OTHERS" : [],
        "AMBIGUOUS": []
    }
    for kernel_no, kernel_name in parsed_stats:
        # print("kernel number: " + str(kernel_no))
        # print("kernel name: " + kernel_name)
        kernel_set = [kernel_no, kernel_name]
        if is_wgrad(kernel_name):
            output["dwCONV"].append(kernel_set)
        elif is_dgrad(kernel_name):
            output["dxCONV"].append(kernel_set)
        elif is_conv(kernel_name):
            output["CONV"].append(kernel_set)
        elif is_dwfc(kernel_name):
            output["dwFC"].append(kernel_set)
        elif is_dxfc(kernel_name):
            output["dxFC"].append(kernel_set)
        elif is_fc(kernel_name):
            output["FC"].append(kernel_set)
        elif is_others(kernel_name):
            output["OTHERS"].append(kernel_set)
        elif is_xla_kernel(kernel_name):
            if (kernel_name.find("fusion") != -1):
                # see fused xla_hlo
                try:
                    kernel_name = "fusion_" + kernel_name.split("_")[1]
                except:
                    kernel_name = "fusion"
                if kernel_name not in metadata_table:
                    # TODO: ???
                    output["BN & ELEMWISE OP"].append(kernel_set)
                    continue
                is_found = False
                for instr in metadata_table[kernel_name]["instr"]:
                    if (instr.find("reduce-window") != -1):
                        output["POOL"].append(kernel_set)
                        is_found = True
                        break
                if is_found:
                    continue
                for data in metadata_table[kernel_name]["data"]:
                    if (data.find("BatchNorm") != -1):
                        output["BN & ELEMWISE OP"].append(kernel_set)
                        break
            else:
                output["BN & ELEMWISE OP"].append(kernel_set)
        else:
            output["AMBIGUOUS"].append(kernel_set)
    hlo_file.close()

    print("AMBIGUOUS!")
    for item in output["AMBIGUOUS"]:
        print("kernel number: " + str(item[0]))
        print("kernel name: " + item[1])

    # return
    outfile = open(args.output_name, "w+")
    json.dump(output, outfile)