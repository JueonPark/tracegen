from xmlrpc.client import MAXINT


def parse_stats(stats_string, start_no, end_no):
  # skip first line
  stats_string_list = stats_string.split("\n")[1:]
  result = []
  first_line_no = start_no
  if end_no < 0:
    end_no = MAXINT
  for line in stats_string_list:
    if line == "":
      continue
    line_parsed = line.split(', ')
    kernel_no = int(line_parsed[0].split('-')[1].split('.')[0])
    kernel_name = line_parsed[1]
    if kernel_no >= start_no and kernel_no <= end_no:
      result.append((kernel_no, kernel_name))
  return result

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--path", type=str, required=True)
  args = parser.parse_args()
  stats_file = open(args.path, 'r')
  result = parse_stats(stats_file.read(), -1, -1)
  print(result)
