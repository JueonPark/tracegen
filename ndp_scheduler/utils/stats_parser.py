def parse_stats(stats_string):
  # skip first line
  stats_string_list = stats_string.split("\n")[1:]
  result = []
  first_line_no = -1
  for line in stats_string_list:
    if line == "":
      continue
    line_parsed = line.split(', ')
    kernel_no = int(line_parsed[0].split('-')[1].split('.')[0])
    kernel_name = line_parsed[1]
    if ('comparison' in kernel_name):
      first_line_no = len(result) + 1
    result.append((kernel_no, kernel_name))
  return result[first_line_no:]

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--path", type=str, required=True)
  args = parser.parse_args()
  stats_file = open(args.path, 'r')
  result = parse_stats(stats_file.read())
  print(result)
