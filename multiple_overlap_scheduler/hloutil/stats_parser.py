from xmlrpc.client import MAXINT


def parse_stats(stats_string, start_no, end_no):
  # skip first line
  stats_string_list = stats_string.split("\n")[1:]
  result = []
  if end_no < 0:
    end_no = MAXINT
  for line in stats_string_list:
    if line == "":
      continue
    line_parsed = line.split(', ')
    try:
      kernel_num = int(line_parsed[0].split('-')[1].split('.')[0])
    except:
      continue
    kernel_name = line_parsed[1]
    if kernel_num >= start_no and kernel_num <= end_no:
      result.append((kernel_name, kernel_num))
  return result

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--path", type=str, required=True)
  args = parser.parse_args()
  stats_file = open(args.path, 'r')
  result = parse_stats(stats_file.read(), -1, -1)
  print(result)
