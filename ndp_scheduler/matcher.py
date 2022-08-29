'''
ts_parsed must be a result from `parse_thunk_schedule`
stats_parsed must be a result from `parse_stats`
'''
def match(ts_parsed, stats_parsed):
  while True:
    if ts_parsed[0] == stats_parsed[0][1]:
      break
    ts_parsed = ts_parsed[1:]

  ts_matched = []
  stats_matched = []
  ts_unmatched = [i for i in ts_parsed]
  stats_unmatched = [i for i in stats_parsed]

  for thunk_name in ts_parsed:
    stats_matched_chunk = []
    for kernel_no, kernel_name in stats_parsed:
      if thunk_name == kernel_name:
        ts_unmatched.remove(thunk_name)
        stats_unmatched.remove((kernel_no, kernel_name))
        ts_matched.append(thunk_name)
        stats_matched_chunk.append((kernel_no, kernel_name))
      elif '__' in kernel_name and thunk_name == kernel_name[:kernel_name.find('__')]:
        stats_unmatched.remove((kernel_no, kernel_name))
        stats_matched_chunk.append((kernel_no, kernel_name))
    if len(stats_matched_chunk) > 0:
      stats_matched.append(stats_matched_chunk)

  return (ts_matched, stats_matched), (ts_unmatched, stats_unmatched)

if __name__ == "__main__":
  import argparse
  from utils import parse_thunk_schedule
  from utils import parse_stats

  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--ts-path", type=str, required=True)
  parser.add_argument("-s", "--stats-path", type=str, required=True)
  args = parser.parse_args()

  ts_parsed = parse_thunk_schedule(open(args.ts_path).read())
  stats_parsed = parse_stats(open(args.stats_path).read())
  (ts_matched, stats_matched), (ts_unmatched, stats_unmatched) = match(ts_parsed, stats_parsed)
  print(ts_matched)
  print(stats_matched)
