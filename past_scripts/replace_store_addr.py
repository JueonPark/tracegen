"""
replace address that 'writes' to CXL memory.
i.e. the kernel that is related to _ON_THE_FLY_ phase 1.
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', required=True, help="file to replace addr")
args = parser.parse_args()

print(args.filename)

# input file
fin = open(args.filename, "rt")
# output file to write the result to
fout = open(args.filename+".out", "wt")
# for each line in the input file
for line in fin:
    # read replace the string and write to output file
    new_line = line
    if "STG" in line:
        new_line = line.replace('0x7', '0x1007')
    fout.write(new_line)
# close input and output files
fin.close()
fout.close()

os.system("mv %s %s"%(args.filename+".out", args.filename))
