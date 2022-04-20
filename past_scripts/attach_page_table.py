"""
attach page table to kernel
this is needed when data is loaded from ndpx to gpu
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trace", required=True, help="traceg file to put page table")
parser.add_argument("--page_table", required=True, help="page table to put")
args = parser.parse_args()

print(args.trace)
print(args.page_table)

filenames = [args.page_table, args.trace]
with open(args.trace + ".out", 'w+') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
    outfile.close()

os.system("mv %s.out %s"%(args.trace, args.trace))