import os
import argparse

from utils import parse_thunk_schedule
from utils import parse_stats
from matcher import match

def topk(l, k=10):
  result = ""
  k = min(k, len(l))
  for i in range(1, k+1):
      result += f"{i}. {l[i-1]}\n"
  return result
    
parser = argparse.ArgumentParser()
parser.add_argument('-S', '--save', type=str, help="If given, this program will save input history", default=None)
parser.add_argument('-t', '--ts-path', type=str, help="module*.thunk_schedule file path", required=True)
parser.add_argument('-s', '--stats-path', type=str, help="stats.csv file path", required=True)
parser.add_argument('-o', '--output', type=str, help="output file name", required=True)
args = parser.parse_args()
if args.save is not None:
  input_history = []
else:
  print("If you are trying to schedule new model, please delete the `input_history.temp` file and retry with the argument `--save True`.")
  print("Existing `input_history.temp` may be generated for another model.")
  print("Without `--save` argument, this program tries previous user input.")
  input_history_file = open('input_history.temp', 'r')
  input_history = input_history_file.read().split('\n')[:-1]
  input_history = [int(i) for i in input_history]

ts_parsed, omitted = parse_thunk_schedule(open(args.ts_path).read())
stats_parsed = parse_stats(open(args.stats_path).read())
(ts_matched, stats_matched), (ts_unmatched, stats_unmatched) = match(ts_parsed, stats_parsed)

for thunk_name in ts_unmatched:
  if len(stats_unmatched) == 0:
    break

  if args.save is not None:
    print(topk(stats_unmatched))
    k = input(f'Determine the last kernel of {thunk_name}: ')
    if k == "quit":
        break
    if k == "no":
        continue
    k = int(k)
    input_history.append(k)
  else:
    k = input_history[0]
    input_history = input_history[1:]

  ts_matched.append(thunk_name)
  print(stats_unmatched[:0])
  stats_matched.append(stats_unmatched[:k])
  stats_unmatched = stats_unmatched[k:]

if args.save is not None:
  input_history_file = open('input_history.temp', 'w')
  for i in input_history:
    input_history_file.write(str(i) + '\n')
        
matched = list(zip(ts_matched, stats_matched))
print(matched)
matched.sort(key=lambda x: x[1][0][0])  # error with sorting...?
print(f'THUNKS: {len(ts_parsed)}')
print(f'STATS:  {len(stats_parsed)}')
print(f'TOTAL:  {len(matched)}')

with open(args.output, 'w') as f:
  for thunk_name, kernels in matched:
    for kernel_no, kernel_name in kernels:
      f.write(f'// Thunk: {thunk_name}\n')
      f.write(f'// Kernel Name: {kernel_name}\n')
      for optim in omitted:
        if thunk_name.split(':')[0] in optim.replace('-', '_').replace('.', '_'):
          f.write(f'// Optim: {optim}\n')
      f.write(f'kernel-{kernel_no}.traceg' + '\n')
      f.write('\n')
