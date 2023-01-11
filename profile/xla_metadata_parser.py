def parse_bert_metadata(input):
  # if (input.find("gradient_tape") != -1):
  #   return "backward"
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return "Loss"
  elif (input.find("bert_pretrain_loss_and_metric_layer") != -1):
    return "Loss"
  elif (input.find("layer_norm") != -1):
    return "LN"
  elif (input.find("LayerNorm") != -1):
    return "LN"
  elif (input.find("layernorm") != -1):
    return "LN"
  elif (input.find("batch_norm") != -1):
    return "LN"
  elif (input.find("BatchNorm") != -1):
    return "LN"
  elif (input.find("batchnorm") != -1):
    return "LN"
  elif (input.find("dropout") != -1):
    return "Dropout"
  elif (input.find("Softmax") != -1):
    return "Softmax"
  elif (input.find("Gelu") != -1):
    return "Gelu"
  elif (input.find("Adam") != -1):
    return "Adam"
  elif (input.find("ResourceApplyAdam") != -1):
    return "Adam"
  elif (input.find("strided_slice") != -1):
    return "EncoderOutput"
  else:
    return "Others"

def parse_cnn_metadata(input):
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return "Loss"
  elif (input.find("bert_pretrain_loss_and_metric_layer") != -1):
    return "Loss"
  elif (input.find("FusedBatchNormV3") != -1):
    return "BN"
  elif (input.find("batch_norm") != -1):
    return "BN"
  elif (input.find("BatchNorm") != -1):
    return "BN"
  elif (input.find("batchnorm") != -1):
    return "BN"
  elif (input.find("Relu6") != -1):
    return "Relu6"
  elif (input.find("Relu") != -1):
    return "Relu"
  elif (input.find("Adam") != -1):
    return "Adam"
  else:
    return "Others"

def parse_vit_metadata(input):
  # if (input.find("gradient_tape") != -1):
  #   return "backward"
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return "Loss"
  elif (input.find("multi_head_self_attention") != -1):
    return "MultiHeadSelfAttention"
  elif (input.find("layer_norm") != -1):
    return "LN"
  elif (input.find("layer_normalization") != -1):
    return "LN"
  elif (input.find("LayerNorm") != -1):
    return "LN"
  elif (input.find("layernorm") != -1):
    return "LN"
  elif (input.find("batch_norm") != -1):
    return "LN"
  elif (input.find("BatchNorm") != -1):
    return "LN"
  elif (input.find("batchnorm") != -1):
    return "LN"
  elif (input.find("transpose") != -1):
    return "Transpose"
  elif (input.find("dropout") != -1):
    return "Dropout"
  elif (input.find("Relu") != -1):
    return "Relu"
  elif (input.find("Adam") != -1):
    return "Adam"
  else:
    return "Others"

def parse_transformer_metadata(input):
  # if (input.find("gradient_tape") != -1):
  #   return "backward"
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return "Loss"
  elif (input.find("multi_head_self_attention") != -1):
    return "MultiHeadSelfAttention"
  elif (input.find("layer_norm") != -1):
    return "LN"
  elif (input.find("layer_normalization") != -1):
    return "LN"
  elif (input.find("LayerNorm") != -1):
    return "LN"
  elif (input.find("layernorm") != -1):
    return "LN"
  elif (input.find("layer_normalization") != -1):
    return "LN"
  elif (input.find("layer_normalization_") != -1):
    return "LN"
  elif (input.find("batch_norm") != -1):
    return "LN"
  elif (input.find("BatchNorm") != -1):
    return "LN"
  elif (input.find("batchnorm") != -1):
    return "LN"
  elif (input.find("transpose") != -1):
    return "Transpose"
  elif (input.find("dropout") != -1):
    return "Dropout"
  elif (input.find("Relu") != -1):
    return "Relu"
  elif (input.find("activation") != -1):
    return "Relu"
  elif (input.find("reduce") != -1):
    return "Reduction"
  elif (input.find("Reduce") != -1):
    return "Reduction"
  elif (input.find("sum") != -1):
    return "Elementwise"
  elif (input.find("Sum") != -1):
    return "Elementwise"
  elif (input.find("add") != -1):
    return "Elementwise"
  elif (input.find("div") != -1):
    return "Elementwise"
  elif (input.find("max") != -1):
    return "Elementwise"
  elif (input.find("Max") != -1):
    return "Elementwise"
  elif (input.find("AssignAddVariableOp") != -1):
    return "Elementwise"
  elif (input.find("ResourceApplyAdam") != -1):
    return "Adam"
  elif (input.find("Adam") != -1):
    return "Adam"
  elif (input.find("time_distributed") != -1):
    return "time_distributed"
  else:
    return "Others"

def parse_dlrm_metadata(input):
  # if (input.find("gradient_tape") != -1):
  #   return "backward"
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return "Loss"
  elif (input.find("softmax_cross_entropy_with_logits") != -1):
    return "Loss"
  elif (input.find("ResourceApplyAdam") != -1):
    return "Adam"
  elif (input.find("Adam") != -1):
    return "Adam"
  elif (input.find("AssignAddVariableOp") != -1):
    return "Add"
  elif (input.find("Relu") != -1):
    return "Relu"
  elif (input.find("embedding_3") != -1):
    return "Embedding"
  elif (input.find("AssignAddVariableOp") != -1):
    return "Elementwise"
  elif (input.find("BiasAdd") != -1):
    return "Elementwise"
  elif (input.find("binary_crossentropy") != -1):
    return "Loss"
  elif (input.find("logistic_loss") != -1):
    return "Loss"
  elif (input.find("logistic_loss") != -1):
    return "Loss"
  elif (input.find("concatenate_1") != -1):
    return "concatenate"
  elif (input.find("div_no_nan_1") != -1):
    return "Elementwise"
  else:
    return "Others"

def parse_lstm_metadata(input):
  # if (input.find("gradient_tape") != -1):
  #   return "backward"
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return "Loss"
  elif (input.find("softmax_cross_entropy_with_logits") != -1):
    return "Loss"
  elif (input.find("ResourceApplyAdam") != -1):
    return "Adam"
  elif (input.find("Adam") != -1):
    return "Adam"
  elif (input.find("AssignAddVariableOp") != -1):
    return "Add"
  elif (input.find("Relu") != -1):
    return "Relu"
  elif (input.find("Sigmoid") != -1):
    return "Sigmoid"
  elif (input.find("sigmoid") != -1):
    return "Sigmoid"
  else:
    return "Others"