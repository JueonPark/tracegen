# merge stats.csv and xla hlo graph(which is txt file)
# output file: stats.csv에서 각 커널이 수행하는 operation이 뭔지 mapping 해준다.
# kernel number, 
import os
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--stats", required=True, help="stats.csv file")
parser.add_argument("--start", required=False, help="starting number of stats.csv file")
parser.add_argument("--end", required=False, help="finising number of stats.csv file")
parser.add_argument("--hlo", required=True, help="hlo graph file")
parser.add_argument("--name", required=False, help="output file name")
args = parser.parse_args()

print(args.stats)
print(args.start)
print(args.end)
print(args.hlo)
print(args.name)

stats = open(args.stats, newline="")
hlo = open(args.hlo)
stats_reader = csv.DictReader(stats)
for row in stats_reader:
    to_search = row[1]
    # e.g. fusion_8 from stats.csv->fused_computation.8
    # e.g. reduce_51->reduce.51
    if (to_search.find("fusion") == -1):
        to_search.replace("_", ".")
    else:
        fusion_num = to_search.split("_")[1]
        to_search = "fused_computation_{%s} "%fusion_num
    