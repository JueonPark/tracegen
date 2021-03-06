# parses op names from HLO instruction.
def parse_metadata(input):
  status = ""
  if (input.find("gradient_tape") != -1):
    status = "backward_"
  elif (input.find("Layer") != -1):
    return input
  elif (len(input) > 0):
    status = ""
  else:
    return "Others"
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return status + "Loss"
  elif (input.find("loss") != -1):
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
  elif (input.find("Cast") != -1):
    return status + "Cast"
  elif (input.find("Adam") != -1):
    return status + "Adam"
  elif (input.find("bert_pretrain_loss_and_metric_layer") != -1):
    return status + "bert_pretrain_loss_and_metric_layer"
  elif (input.find("predictions") != -1):
    return status + "predictions"
  elif (input.find("pooler_transform") != -1):
    return status + "pooler_transform"
  elif (input.find("strided_slice") !=- -1):
    return status + "strided_slice"
  elif (input.find("type_embedding") !=- -1):
    return status + "type_embedding"
  elif (input.find("position_embedding") !=- -1):
    return status + "position_embedding"
  elif (input.find("word_embedding") !=- -1):
    return status + "word_embedding"
  elif (input.find("layer_0") != -1):
    try:
      return status + input.split("layer_0/")[1]
    except:
      return status + input
  else:
    return status + input