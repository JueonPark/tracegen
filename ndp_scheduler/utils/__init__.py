from .thunk_schedule_parser import parse_thunk_schedule
from .stats_parser import parse_stats
from .schedule_parser import parse_schedule
from .schedule_parser import dump_schedule
from .hlo_parser import parse_hlo

CONV_DGRAD_LIST = [
    ['s1688cudnn', 'dgrad'],
    ['dgrad_1x1_stride_2x2'],
    ['dgrad2d_c1_k1'],
]

CONV_WGRAD_LIST = [
    ['s1688cudnn', 'wgrad'],
    ['wgrad2d_shmem_tiling_kernel'],
    ['wgrad2d_c1_k1_nhwc_kernel'],
    ['wgrad_alg0_engine'],
    ['first_layer_wgrad_kernel']
]

CONV_BW_LIST = [
    ['s1688cudnn'],
    ['s884cudnn']
]

GEMM_BW_LIST = [
    ['gemmSN_TN_kernel'],
    ['s1688gemm'],
    ['s884gemm']
]

def type_kernel_bw(kernel_name):
    for conv_dgrad_conjunctive in CONV_DGRAD_LIST:
        if all([(i in kernel_name) for i in conv_dgrad_conjunctive]):
            return 'conv_dgrad'

    for conv_wgrad_conjunctive in CONV_WGRAD_LIST:
        if all([(i in kernel_name) for i in conv_wgrad_conjunctive]):
            return 'conv_wgrad'

    for gemm_conjunctive in GEMM_BW_LIST:
        if all([(i in kernel_name) for i in gemm_conjunctive]):
            return 'gemm'

    for conv_conjunctive in CONV_BW_LIST:
        if all([(i in kernel_name) for i in conv_conjunctive]):
            return 'conv'

    return None
