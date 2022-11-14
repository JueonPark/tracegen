'''
thunk_schedule expects ts_string to be a raw string
from thunk_schedule file.
'''
def parse_thunk_schedule(ts_string):
  GPU_thunks = []
  NDP_thunks = []
  ts_string_list = ts_string.split('\n')
  order = 0
  for line in ts_string_list:
    if not line.startswith('k'):
      continue # skip non-kernel line

    line_parsed = line.split('\t')
    if len(line_parsed) <= 1:
      continue
    kernel_name = line_parsed[1].split(' ')[0]
    if 'Ndp' in kernel_name:
      NDP_thunks.append((order,kernel_name.replace('%','')))
      order+=1
    elif 'copy' in kernel_name:
      continue
    else: 
      GPU_thunks.append((order, kernel_name.replace('%','')))
      order+=1
  return GPU_thunks, NDP_thunks

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--path", type=str, required=True)
  args = parser.parse_args()
  ts_file = open(args.path, 'r')
  print(parse_thunk_schedule(ts_file.read()))

