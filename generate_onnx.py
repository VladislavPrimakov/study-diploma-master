from networks import EmbeddingNet, ConvolutionalNet, SiameseNetSigmoid
import torch
from torch import nn


# torch.onnx.export(ConvolutionalNet(1, 10), torch.randn(1, 1, 28, 28) , "model.onnx", input_names=["MNIST"], output_names=["Class probabilities"])


# torch.onnx.export(SiameseNetSigmoid(1), torch.randn(1, 2048) , "model.onnx", input_names=["MNIST"], output_names=["Predict"])


torch.onnx.export(EmbeddingNet(1), torch.randn(1, 1, 28, 28) , "model.onnx", input_names=["MNIST"], output_names=["Embedding"])

