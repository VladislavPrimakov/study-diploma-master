from networks import EmbeddingNet, ConvolutionalNet
import torch
from torch import nn



torch.onnx.export(ConvolutionalNet(1, 10), torch.randn(1, 1, 28, 28) , "model.onnx", input_names=["MNIST"], output_names=["Class probabilities"])
