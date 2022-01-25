import csv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--stats', required=True, help="stats.csv file")
args = parser.parse_args()

print(args.stats)

table = []

stats = open(args.stats)
stats_reader = csv.reader(stats)
for row in stats_reader:
    to_search = row[1]
    if to_search in table:
        pass
    else:
        table.append(to_search)

result = open(args.stats+".out", "w+")
for item in table:
    if (item.find("fusion") != -1):
        continue
    elif (item.find("add") != -1):
        continue
    elif (item.find("reduce") != -1):
        continue
    elif (item.find("convert") != -1):
        continue
    elif (item.find("log") != -1):
        continue
    result.write(item + "\n")
result.close()