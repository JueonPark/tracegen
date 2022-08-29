"""
parse hlo string
fusion -> 
return metadata in hlo text indexed by HLO op name
"""

class HloTable(object):
  def __init__(self, hlo_string):
    self.hlo_table = dict()
    splitted_hlo_string = hlo_string.split('\n\n')
    for computation in splitted_hlo_string:
      metadata = []
      lines = computation.split('\n')
      if not 'fused_computation' in lines[0]:
        # hlo_table only contains fusions
        continue
      start = lines[0].find('%')
      end = lines[0].find(' (')
      fused_computation = lines[0][start:end]
      try:
        fusion_no = fused_computation.split('.')[1]
        fusion_string = f'fusion_{fusion_no}'
      except:
        fusion_string = 'fusion'
      # print("fused_computation: " + fusion_string)
      self.hlo_table[fusion_string] = {
        "instr" : [],
        "data"  : []
      }
      for line in lines:
        # print(line)
        start = line.find('%')
        end = line.find('=')
        if end == -1:
          continue
        op_name = line[start:end-1]
        self.hlo_table[fusion_string]["instr"].append(op_name)
        start = line.find("op_name=")
        if (start == -1):
          # print("metadata not found")
          continue
        # print(line[start+9:-2])
        self.hlo_table[fusion_string]["data"].append(line[start+9:-2])


def parse_hlo(hlo_string):
  return HloTable(hlo_string)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--hlo-path', type=str, required=True)
  args = parser.parse_args()

  parse_hlo(open(args.hlo_path, 'r').read())