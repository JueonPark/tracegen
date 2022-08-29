'''
thunk_schedule expects ts_string to be a raw string
from thunk_schedule file.
'''
def parse_thunk_schedule(ts_string):
  result = []
  omitted_result = []
  ts_string_list = ts_string.split('\n')
  for line in ts_string_list:
    if not line.startswith('k'):
      continue # skip non-kernel line

    line_parsed = line.split('\t')
    if len(line_parsed) <= 1:
      continue
    kernel_name = line_parsed[1].split(' ')[0]
    if 'copy' in kernel_name:
      continue
    if 'NdpAdam' in kernel_name:
      omitted_result.append(kernel_name)
      continue
    result.append(kernel_name.replace(".", "_").replace("-", "_"))
  return result, omitted_result

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--path", type=str, required=True)
  args = parser.parse_args()
  ts_file = open(args.path, 'r')
  print(parse_thunk_schedule(ts_file.read()))

