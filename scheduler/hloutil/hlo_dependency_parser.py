"""
Parse hlo graph for dependency check
"""

from opcode import opname
from posixpath import split
from typing import Set

# HloDepdendencyManager has 5 members
# - hlo_table
# - depend_table
# - hlo_hops
# - metadata_table
# - custom call table for custom call information
class HloDepdendencyManager(object):
  def __init__(self, hlo_string):
    self.original_hlo_table = dict()
    self.hlo_table = dict()
    self.custom_call_table = dict()
    self.metadata_table = dict() # metadata table is used to find the fw/bw boundary
    splitted_hlo_string = hlo_string.split('\n\n')
    xla_hlo = ''
    for computing in splitted_hlo_string:
      if 'ENTRY %cluster' in computing:
        xla_hlo = computing
        break
    lines = xla_hlo.split('\n')
    for line in lines:
      start = line.find('%')
      end = line.find('=')
      metadata_start = line.find('op_name="')
      metadata_end = line.find('"', metadata_start+10)
      custom_call_target_start = line.find('custom_call_target="')
      custom_call_target_end = line.find('"', custom_call_target_start+20)
      if end == -1:
        continue
      op_name = line[start+1:end-1]
      self.original_hlo_table[op_name] = line
      # ignore argument or constant
      if 'arg' in op_name or 'constant' in op_name:
        continue
      params = line.split('=')[1].split(op_name.split('.')[0]+'(')[1].split('),')[0]
      params = self.parse_params(params)
      self.hlo_table[op_name] = params
      # if there's custom call, add the custom call target to custom_call_table
      if custom_call_target_start != -1:
        custom_call_target = line[custom_call_target_start+20:custom_call_target_end]
        self.custom_call_table[op_name] = custom_call_target
      if metadata_start == -1:
        if op_name.find("fusion") != -1:
          fused_computation_start = line.find(', calls=%')
          fused_computation = line[fused_computation_start+9:]
          computation_to_find = ''
          for computing in splitted_hlo_string:
            if fused_computation in computing:
              computation_to_find = computing
              break
          new_lines = computation_to_find.split('\n')
          for new_line in new_lines:
            metadata_start = new_line.find('op_name="')
            metadata_end = new_line.find('"', metadata_start+10)
            if metadata_start == -1:
              continue
            metadata = new_line[metadata_start+9:metadata_end]
            self.metadata_table[op_name] = metadata
      else:
        metadata = line[metadata_start+9:metadata_end]
        self.metadata_table[op_name] = metadata
      
    
    # generate other dictionaries
    self.depend_table = dict()
    self.hlo_hops = dict()
    for key in self.hlo_table:
      self.depend_table[key] = self.get_all_dependent_kernels(key)
      self.hlo_hops[key] = self.get_custom_call_hops(key)
    # print(self.hlo_hops)
    pass

  def parse_params(self, params):
    result = []
    start = params.find('%')
    end = params.find(', ')
    if(start == -1):
      return []
    splited = params[start+1:].split(', ')
    arg = splited[0]
    if not 'arg' in arg and not 'constant' in arg:
      result.append(arg.replace(')', ''))
    if len(splited) > 1:
      result += self.parse_params(params[end+1:])
    return result

  def is_dependent(self, tester, base):
    # if tester == base:
    #   return True
    # result = False
    # operands = self.hlo_table[tester]
    # for operand in operands:
    #   result = result or self.is_dependent(operand, base)
    return base in self.depend_table[tester]
  
  def get_all_dependent_kernels(self, name):
    operands = self.hlo_table[name]
    result = set()
    for operand in operands:
      if not 'get-tuple-element' in operand and not 'bitcast' in operand :
        result.add(operand)
      if operand in self.depend_table:
        result = result | self.depend_table[operand]
      else:
        result = result | self.get_all_dependent_kernels(operand)
    return result
  
  def can_launch(self, name, finish_list):
    dependent_kernels = self.depend_table[name]
    remainings = dependent_kernels.copy()
    for depend in dependent_kernels:
      if depend in finish_list:
        remainings.remove(depend)
    self.depend_table[name] = remainings
    return len(remainings) == 0
  
  def print_unfinished_parent(self, name):
    # print(self.depend_table[name])
    pass
  
  def get_custom_call_hops(self, name):
    if name in self.hlo_hops:
      return self.hlo_hops[name]
    result = 0
    # custom call target을 봐서 "wise"가 있으면 넘어가도록 하자.
    is_custom_call = 0
    if name in self.custom_call_table and \
          self.custom_call_table[name].find("wise") == -1:
      is_custom_call = 1
    result = is_custom_call
    for operand in self.hlo_table[name]:
      result = max(result, self.get_custom_call_hops(operand) + is_custom_call)
    self.hlo_hops[name] = result
    return result

  def print_direct_parent_custom_call(self, name):
    for operand in self.hlo_table[name]:
      if 'custom-call' in operand:
        # print(operand)
        pass
      else:
        self.print_direct_parent_custom_call(operand)
  
  def get_operands(self, name):
    result = []
    for operand in self.hlo_table[name]:
      if 'get-tuple-element' in operand or 'bitcast' in operand:
        result = result + self.get_operands(operand)
      else:
        result.append(operand)
    return result

  def get_hops_map(self):
    return self.hlo_hops.copy()

  def refill_hops_map(self, hops_table, name):
    if name in hops_table:
      return hops_table[name]
    for operand in self.hlo_table[name]:
      hops_table = self.get_custom_call_hops(operand)
    result = 0
    is_custom_call =  1 if 'custom-call' in name else 0
    result = is_custom_call
    for operand in self.hlo_table[name]:
      result = max(result, is_custom_call + hops_table[operand])
    hops_table[name] = result
    return hops_table

  # tester: gpu custom call
  # base: ndp custom call
  def need_bert_ph3_scheduling(self, tester, base):
    # TODO: get-tuple-element가 index 0여야 한다.
    # first, get the operands
    for arg in self.hlo_table[tester]:
      if base in self.original_hlo_table[arg] and "index=0" in self.original_hlo_table[arg]:
        # print(self.original_hlo_table[arg])
        return True
    return False

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--path", type=str, required=True)
  args = parser.parse_args() 
  hlo_file = open(args.path, 'r')
  manager = HloDepdendencyManager(hlo_file.read())
  # print(manager.is_dependent('convert.356', 'custom-call.192'))
  pass
