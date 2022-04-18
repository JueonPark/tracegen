import os
import argparse

from black import is_stub_body

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--packet-size', type=int, required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--buffer', type=int, required=True)
parser.add_argument('--sync', type=bool, default=False, required=False)
parser.add_argument('--simd', type=int, required=True)
# parser.add_argument('--passes', type=str, required=True) # fw, bw or all
args = parser.parse_args()


num_gpu = args.gpu
num_buffer = args.buffer
is_sync = 1
if args.sync == False:
    is_sync = 0

config = f'{args.model}/packet_128_buffer_{num_gpu}_gpu_{num_buffer}_sync_{is_sync}_simd_8'
directions = ['fw', 'bw']
max_memory_access_size = 128

for direction in directions:
    target_config = config + '_' + direction
    print(target_config)
    # assert(False)
    home = './' + target_config + '/'
    kernels = os.listdir(home)
    for kernel in kernels:
        kernel_home = home+kernel+'/'

        # gpu_write_addr start
        gpu = 'GPU_0'
        gpu_dir = kernel_home+gpu+'/'
        kernelslist = open(gpu_dir+'kernelslist.g', 'r')
        scheduled_kernels = kernelslist.readlines()
        kernelslist.close()
        
        require_gpu_write_addr = False
        write_base = 0
        act_gpu_write_addr = ''
        for scheduled in scheduled_kernels:
            if '_ON_' in scheduled and 'ph1' in scheduled:
                if direction == 'bw':
                    assert(False)
                # get gpu write addr
                print(gpu_dir)
                ph1_kernel = open(gpu_dir+scheduled[:-1], 'r')
                ph1_kernel_trace = ph1_kernel.readlines()
                ph1_kernel.close()

                for trace_line in ph1_kernel_trace:
                    if 'SET_FILTER' in trace_line:
                        write_base = int(trace_line.split(' ')[1], 16)
                        act_gpu_write_addr = f'-act_gpu_write_addr = {hex(write_base)}\n'
                        require_gpu_write_addr = True
                        print(act_gpu_write_addr)
                        break

        if require_gpu_write_addr:
            for scheduled in scheduled_kernels:
                if 'kernel-' in scheduled:
                    # write gpu write addr
                    gpu_kernel = open(gpu_dir+scheduled[:-1], 'r')
                    trace_line = gpu_kernel.readlines()
                    gpu_kernel.close()
                    gpu_kernel = open(gpu_dir+scheduled[:-1], 'w')
                    gpu_kernel.write(act_gpu_write_addr)
                    for trace in trace_line:
                        gpu_kernel.write(trace)
                    gpu_kernel.close()
        # gpu_write_addr end

        # gpu_page_table start
        require_act_cxl = False
        for scheduled in scheduled_kernels:
            if '_ON_' in scheduled and 'ph3' in scheduled:
                    if direction == 'bw':
                        assert(False)
                    require_act_cxl = True
                    print(gpu_dir)
        if require_act_cxl:
            for scheduled in scheduled_kernels:
                if 'kernel-' in scheduled:
                # get act_cxl_addr
                    gpu_kernel = open(gpu_dir+scheduled[:-1], 'r')
                    trace_line = gpu_kernel.readlines()
                    gpu_kernel.close()

                    for i, trace in enumerate(trace_line):
                        if 'act_cxl_addr' in trace:
                            words = trace.split(' ')
                            cxl_addr = int(words[-1], 16)
                            m_gpu_write_offset =  0 - (cxl_addr % max_memory_access_size)
                            cxl_addr += m_gpu_write_offset
                            words[-1] = f'{hex(cxl_addr)}\n'
                            trace_line[i] = ' '.join(words)
                            print(trace_line[i])
                            break
                    gpu_kernel = open(gpu_dir+scheduled[:-1], 'w')
                    for trace in trace_line:
                        gpu_kernel.write(trace)
                    gpu_kernel.close()
        # gpu_page_table end

        # align addr
        gpus = os.listdir(kernel_home)
        for gpu in gpus:
            if 'GPU' not in gpu:
                continue
            gpu_dir = kernel_home+gpu+'/'
            kernelslist = open(gpu_dir+'kernelslist.g', 'r')
            scheduled_kernels = kernelslist.readlines()
            kernelslist.close()
            for scheduled in scheduled_kernels:
                if '_NDP_' in scheduled or '_ON_THE' in scheduled:
                    NDP_trace_file = open(gpu_dir+scheduled[:-1], 'r')
                    NDP_trace = NDP_trace_file.readlines()
                    NDP_trace_file.close()

                    NDP_trace_file = open(gpu_dir+scheduled[:-1], 'w')
                    for trace_line in NDP_trace:
                        words = trace_line.split(' ')
                        for i, word in enumerate(words):
                            if '0x' in word:
                                addr = 0
                                if '\n' in word:
                                    addr = int(word[:-1], 16)
                                else:
                                    addr = int(word, 16)
                                if addr == 0:
                                    assert(False)
                                m_gpu_write_offset =  0 - (addr % max_memory_access_size)
                                addr += m_gpu_write_offset
                                if '\n' in word:
                                    words[i] = f'{hex(addr)}\n'
                                else:
                                    words[i] = f'{hex(addr)}'
                                trace_line = ' '.join(words)
                        NDP_trace_file.write(trace_line)
                    NDP_trace_file.close()

                    

