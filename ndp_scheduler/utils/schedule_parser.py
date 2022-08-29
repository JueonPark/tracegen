def parse_schedule(schedule_string):
    kernels = schedule_string.split('\n\n')
    # Thunk, Kernel Name, Optim, GPU Kernel
    results = []
    for kernel in kernels:
        result = {
            'thunk_name': None,
            'kernel_name': None,
            'optims': [],
            'ndp_kernels': [],
            'gpu_kernel': None,
            'no_cxl': False
        }
        if kernel == "":
            continue
        for kernel_string_line in kernel.split('\n'):
            if kernel_string_line == "":
                continue
            if kernel_string_line.startswith('//'):
                if 'Thunk' in kernel_string_line:
                    s = kernel_string_line.find(':') + 2
                    result['thunk_name'] = kernel_string_line[s:]
                elif 'Kernel Name' in kernel_string_line:
                    s = kernel_string_line.find(':') + 2
                    result['kernel_name'] = kernel_string_line[s:]
                elif 'Optim' in kernel_string_line:
                    s = kernel_string_line.find(':') + 2
                    result['optims'].append(kernel_string_line[s:])
            elif kernel_string_line.startswith('kernel-'):
                result['gpu_kernel'] = kernel_string_line
            elif kernel_string_line.startswith('#'):
                if 'NO_CXL' in kernel_string_line:
                    result['no_cxl'] = True
            else:
                raise Exception('Cannot parse string: ' + kernel_string_line)
        results.append(result)
    return results

def dump_schedule(schedule):
    result_string = ""
    for kernel in schedule:
        print(kernel)
        if kernel['thunk_name'] is not None:
            result_string += f'// Thunk: {kernel["thunk_name"]}\n'
        if kernel['kernel_name'] is not None:
            result_string += f'// Kernel Name: {kernel["kernel_name"]}\n'
        if len(kernel['optims']):
            for optim in kernel["optims"]:
                result_string += f'// Optim: {optim}\n'
        if kernel['no_cxl']:
            result_string += f'# NO_CXL\n'
        if len(kernel['ndp_kernels']):
            for ndp_kernel in kernel['ndp_kernels']:
                result_string += f'{ndp_kernel}\n'
        if kernel['gpu_kernel'] is not None:
            result_string += f'{kernel["gpu_kernel"]}\n'
        result_string += '\n'
    return result_string

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule', type=str, required=True)
    args = parser.parse_args()

    schedule = parse_schedule(open(args.schedule, 'r').read())
    for kernel in schedule:
        print(kernel['gpu_kernel'])
    print(len(schedule))

    print(dump_schedule(schedule))
