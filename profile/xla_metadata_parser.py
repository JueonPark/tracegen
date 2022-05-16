# parses op names from HLO instruction.
def parse_metadata(input):
  status = ""
  if (input.find("gradient_tape") != -1):
    status = "backward_"
  elif (len(input) > 0):
    status = ""
  else:
    return "Others"
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return status + "Loss"
  elif (input.find("layer_norm") != -1):
    return status + "LN"
  elif (input.find("LayerNorm") != -1):
    return status + "LN"
  elif (input.find("layernorm") != -1):
    return status + "LN"
  elif (input.find("batch_norm") != -1):
    return status + "LN"
  elif (input.find("BatchNorm") != -1):
    return status + "LN"
  elif (input.find("batchnorm") != -1):
    return status + "LN"
  elif (input.find("dropout") != -1):
    return status + "Dropout"
  elif (input.find("Softmax") != -1):
    return status + "Softmax"
  elif (input.find("BiasAdd") != -1):
    return status + "Biasadd"
  elif (input.find("Gelu") != -1):
    return status + "Gelu"
  elif (input.find("TanhGrad") != -1):
    return status + "TanhGrad"
  else:
    return status + "Others"