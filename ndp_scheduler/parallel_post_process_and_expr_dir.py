import os

models = ['new_resnet18_ndp_batch_64', 'new_resnet50_ndp_batch_64', 'new_mobilenetv2_ndp_batch_64' , 'new_vgg16_ndp_batch_32']
#models = ['new_mobilenetv2_ndp_batch_64' , 'new_vgg16_ndp_batch_32']
#models = ['new_vgg16_ndp_batch_32']
#gpus = [1,2,4]
gpus = [4]

for m in models:
    for g in gpus:
        post_process_cmd = f'python post_process.py --model ../{m} --packet-size 32 --gpu {g} --buffer {g} --simd 8 --passes all --sync 1\n'
        make_expr_dir_cmd = f'python make_expr_dir.py --model ../{m} --packet-size 32 --gpu {g} --buffer {g} --simd 8 --passes all -- sync 1\n'

        with open(f'post_process_{m}_{g}.sh', 'w') as temp_run:
            temp_run.write("#!/bin/bash\n\n")
            temp_run.write(post_process_cmd)
            temp_run.write(make_expr_dir_cmd)
        slurm_cmd = f'sbatch -J {m}_{g} --partition=allcpu,allgpu --ntasks 1 --cpus-per-task 1 -o {m}_{g}.out -e {m}_{g}.err post_process_{m}_{g}.sh'
        os.system(slurm_cmd)

